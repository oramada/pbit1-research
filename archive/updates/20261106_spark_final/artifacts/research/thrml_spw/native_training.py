"""Sampler-aware positive-MNIST fine-tuning with official THRML updates.

The native path uses two static graphs, one per layer. For each matrix,
p=sigmoid(theta) and the unary Ising field is theta/2, so z=(s+1)/2 has mean p.
Layer 0 is sampled and averaged before ReLU; layer 1 is then sampled independently
and averaged using that shared hidden activation. The STE uses expected-linear
derivatives with native preactivation averages from THRML's sampler.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from thrml import (AbstractObserver, Block, BlockGibbsSpec, FactorSamplingProgram,
                   SamplingSchedule, SpinNode, block_state_to_global,
                   from_global_state, sample_states, sample_with_observation)
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional

from .provenance import manifest


N0 = 64 * 32
N1 = 32 * 10
N_WEIGHT = N0 + N1
W0_SHAPE = (64, 32)
W1_SHAPE = (32, 10)
LOWER_P = 1e-4
UPPER_P = 1.0 - LOWER_P


class NetworkPreactivationSum(AbstractObserver):
    readout: object
    inputs: jax.Array
    bias: jax.Array
    weight_shape: tuple

    def init(self):
        return jnp.zeros((self.inputs.shape[0], self.weight_shape[1]), dtype=jnp.float32)

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)
        z = from_global_state(global_state, program.gibbs_spec, [self.readout])[0]
        sampled_weight = z.astype(jnp.float32).reshape(self.weight_shape)
        preactivation = self.inputs @ sampled_weight + self.bias
        return carry + preactivation, None


def static_graph(n_nodes):
    nodes = [SpinNode() for _ in range(n_nodes)]
    readout = Block(nodes)
    free = Block(nodes)
    spec = BlockGibbsSpec([free], [], {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)})
    initial = [jnp.zeros((n_nodes,), dtype=jnp.bool_)]
    return readout, free, spec, initial


def make_program(free, spec, theta):
    unary_fields = theta.reshape(-1) / 2.0
    factor = SpinEBMFactor([free], unary_fields)
    return FactorSamplingProgram(spec, [SpinGibbsConditional()], [factor], [])


def native_sampled_output(theta, key, inputs, bias, weight_shape,
                          readout, free, spec, initial, samples):
    program = make_program(free, spec, theta)
    observer = NetworkPreactivationSum(readout, inputs, bias, weight_shape)
    schedule = SamplingSchedule(n_warmup=1, n_samples=samples, steps_per_sample=1)
    sums, _ = sample_with_observation(
        key, program, schedule, initial, [], observer.init(), observer)
    return sums / samples


def native_sampled_layer_pair(theta0, theta1, key0, key1, inputs, bias0, bias1,
                              readout0, free0, spec0, initial0,
                              readout1, free1, spec1, initial1, samples):
    # Required schedule: accumulate layer 0, activate once, then use that one
    # hidden tensor as fixed input while independently accumulating layer 1.
    pre0 = native_sampled_output(theta0, key0, inputs, bias0, W0_SHAPE,
                                 readout0, free0, spec0, initial0, samples)
    hidden = jax.nn.relu(pre0)
    logits = native_sampled_output(theta1, key1, hidden, bias1, W1_SHAPE,
                                   readout1, free1, spec1, initial1, samples)
    return pre0, logits


def loss_expected(theta0, theta1, bias0, bias1, inputs, labels):
    w0, w1 = jax.nn.sigmoid(theta0), jax.nn.sigmoid(theta1)
    logits = jax.nn.relu(inputs @ w0 + bias0) @ w1 + bias1
    return jnp.mean(-jax.nn.log_softmax(logits)[jnp.arange(labels.shape[0]), labels])


def make_native_loss(layer0, layer1, samples):
    readout0, free0, spec0, initial0 = layer0
    readout1, free1, spec1, initial1 = layer1
    def objective(theta0, theta1, bias0, bias1, inputs, labels, key):
        mean0 = inputs @ jax.nn.sigmoid(theta0) + bias0
        key0, key1 = jax.random.split(key)
        native0 = native_sampled_output(theta0, key0, inputs, bias0, W0_SHAPE,
                                        readout0, free0, spec0, initial0, samples)
        pre0 = mean0 + jax.lax.stop_gradient(native0 - mean0)
        hidden = jax.nn.relu(pre0)
        mean_out = hidden @ jax.nn.sigmoid(theta1) + bias1
        native1 = native_sampled_output(theta1, key1, hidden, bias1, W1_SHAPE,
                                        readout1, free1, spec1, initial1, samples)
        logits = mean_out + jax.lax.stop_gradient(native1 - mean_out)
        log_probs = jax.nn.log_softmax(logits)
        return jnp.mean(-log_probs[jnp.arange(labels.shape[0]), labels])
    return objective


def adam_init(params):
    return [(jnp.zeros_like(p), jnp.zeros_like(p)) for p in params]


def adam_step(params, grads, moments, step, learning_rate):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    new_params, new_moments = [], []
    for p, g, (m, v) in zip(params, grads, moments):
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g*g
        mh = m / (1-beta1**step)
        vh = v / (1-beta2**step)
        p = p - learning_rate * mh / (jnp.sqrt(vh)+eps)
        # Keep every weight a free p-bit and keep its field finite.
        p = jnp.clip(p, np.log(LOWER_P/(1-LOWER_P)), np.log(UPPER_P/(1-UPPER_P)))
        new_params.append(p)
        new_moments.append((m, v))
    return new_params, new_moments


def init_theta(weights):
    clipped = [np.clip(w, LOWER_P, UPPER_P).astype(np.float32) for w in weights]
    return [jnp.asarray(np.log(w)-np.log1p(-w)) for w in clipped], clipped


def initial_arrays(path):
    data = np.load(path)
    weights = [data["weight_0"], data["weight_1"]]
    biases = [data["bias_0"], data["bias_1"]]
    theta, clipped = init_theta(weights)
    return theta, [jnp.asarray(b) for b in biases], clipped, weights, biases


def prepare_batches(train_x, train_y, seed, batch_size):
    # Order is fixed before reading/constructing trainable logits.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_y))
    batches = []
    for start in range(0, len(order), batch_size):
        ids = order[start:start+batch_size]
        if len(ids) < batch_size:
            ids = np.concatenate([ids, order[:batch_size-len(ids)]])
        batches.append((jnp.asarray(train_x[ids]), jnp.asarray(train_y[ids], dtype=jnp.int32)))
    return batches, order


def validation_metrics(theta, biases, x, y, batch_size):
    w0, w1 = jax.nn.sigmoid(theta[0]), jax.nn.sigmoid(theta[1])
    correct = total = ce_sum = 0.0
    for start in range(0, len(y), batch_size):
        xb = jnp.asarray(x[start:start+batch_size])
        yb = jnp.asarray(y[start:start+batch_size], dtype=jnp.int32)
        if xb.shape[0] == 0:
            continue
        logits = jax.nn.relu(xb @ w0 + biases[0]) @ w1 + biases[1]
        labels = np.asarray(yb)
        logits_np = np.asarray(logits)
        correct += float(np.sum(logits_np.argmax(axis=1) == labels))
        ce_sum += float(np.sum(logsumexp(logits_np, axis=1)-logits_np[np.arange(len(labels)), labels]))
        total += len(labels)
    return {"accuracy": correct/total, "cross_entropy": ce_sum/total}


def native_validation(theta, biases, x, y, seed, samples, layer0, layer1, batch_size):
    correct = total = ce_sum = 0.0
    for b, start in enumerate(range(0, len(y), batch_size)):
        xb = x[start:start+batch_size]
        yb = y[start:start+batch_size]
        true_len = len(yb)
        if true_len < batch_size:
            pad = batch_size-true_len
            xb = np.concatenate([xb, np.repeat(xb[-1:], pad, axis=0)])
            yb = np.concatenate([yb, np.repeat(yb[-1:], pad)])
        key = jax.random.fold_in(jax.random.key(seed), b)
        key0, key1 = jax.random.split(key)
        pre0, logits = native_sampled_layer_pair(
            theta[0], theta[1], key0, key1, jnp.asarray(xb), biases[0], biases[1],
            *layer0, *layer1, samples)
        logits = np.asarray(logits)[:true_len]
        yb = yb[:true_len]
        correct += int(np.sum(logits.argmax(axis=1) == yb))
        ce_sum += float(np.sum(logsumexp(logits, axis=1)-logits[np.arange(true_len), yb]))
        total += true_len
    return {"accuracy": correct/total, "cross_entropy": ce_sum/total}


def train_condition(kind, seed, epochs, native_samples, batch_size, learning_rate,
                    arrays, layer0, layer1):
    train_x, train_y, val_x, val_y = arrays
    batches, order = prepare_batches(train_x, train_y, seed, batch_size)
    # Initialize only after the seeded minibatch order has been fixed.
    theta, biases, clipped, original_weights, original_biases = initial_arrays(
        Path("data/thrml_spw/positive_mnist")/f"seed{seed}.npz")
    losses, epoch_metrics = [], []
    if kind == "expected_control":
        grad_fn = jax.jit(jax.value_and_grad(loss_expected, argnums=(0, 1)))
    elif kind == "native_thrml":
        objective = make_native_loss(layer0, layer1, native_samples)
        grad_fn = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    else:
        raise ValueError(kind)
    moments = adam_init(theta)
    global_step = 0
    val_expected = validation_metrics(theta, biases, val_x, val_y, batch_size)
    val_native = native_validation(theta, biases, val_x, val_y, seed+20000,
                                   native_samples, layer0, layer1, batch_size)
    for epoch in range(epochs):
        total_loss = 0.0
        for minibatch, (xb, yb) in enumerate(batches):
            global_step += 1
            if kind == "expected_control":
                loss, grads = grad_fn(theta[0], theta[1], biases[0], biases[1], xb, yb)
            else:
                key = jax.random.fold_in(jax.random.key(seed+30000+epoch), minibatch)
                loss, grads = grad_fn(theta[0], theta[1], biases[0], biases[1], xb, yb, key)
            theta, moments = adam_step(theta, grads, moments, global_step, learning_rate)
            total_loss += float(loss)*len(yb)
        train_loss = total_loss / (len(batches)*batch_size)
        losses.append(train_loss)
        val_expected = validation_metrics(theta, biases, val_x, val_y, batch_size)
        val_native = native_validation(theta, biases, val_x, val_y, seed+20000+epoch,
                                       native_samples, layer0, layer1, batch_size)
        epoch_metrics.append({"epoch": epoch+1, "train_loss": train_loss,
                              "expected_validation": val_expected,
                              "native_validation": val_native})
        print(f"seed={seed} method={kind} epoch={epoch+1} trainCE={train_loss:.4f} "
              f"val_expected={val_expected['accuracy']:.4f} val_native={val_native['accuracy']:.4f}",
              flush=True)
    return {"theta": theta, "biases": biases, "clipped_weights": clipped,
            "original_weights": original_weights, "original_biases": original_biases,
            "losses": losses, "epoch_metrics": epoch_metrics,
            "batch_order": order}


def export_weights(path, theta, biases):
    p = [np.asarray(jax.nn.sigmoid(v), dtype=np.float32) for v in theta]
    b = [np.asarray(v, dtype=np.float32) for v in biases]
    np.savez_compressed(path, weight_0=p[0], weight_1=p[1], bias_0=b[0], bias_1=b[1])
    return p, b


def evaluate_test_native(theta, biases, data, seed, samples,
                         layer0, layer1, batch_size, outdir, name):
    x, y = data
    output=[]
    for b,start in enumerate(range(0,len(y),batch_size)):
        xb=x[start:start+batch_size];yb=y[start:start+batch_size];n=len(yb)
        if n<batch_size:
            pad=batch_size-n
            xb=np.concatenate([xb,np.repeat(xb[-1:],pad,axis=0)])
        key=jax.random.fold_in(jax.random.key(seed),b)
        key0,key1=jax.random.split(key)
        _,logits=native_sampled_layer_pair(theta[0],theta[1],key0,key1,jnp.asarray(xb),
            biases[0],biases[1],*layer0,*layer1,samples)
        output.append(np.asarray(logits)[:n])
    logits=np.concatenate(output)
    np.savez_compressed(outdir/f"{name}_native_test.npz",logits=logits,labels=y)
    return {"test_accuracy":float(np.mean(logits.argmax(1)==y)),
            "test_cross_entropy":float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y]))}


def explicit_layer_samples(theta, key, readout, free, spec, initial, samples):
    """Test helper: sample THRML states, then explicitly reconstruct sampled matrices."""
    program = make_program(free, spec, theta)
    schedule = SamplingSchedule(n_warmup=1, n_samples=samples, steps_per_sample=1)
    states = sample_states(key, program, schedule, initial, [], [readout])[0]
    return np.asarray(states, dtype=np.float32)


def native_schedule_self_test():
    """Check observers against explicit THRML sample_states matrix reconstruction."""
    data = np.load("data/thrml_spw/positive_mnist/evaluation_data.npz")
    checkpoint = np.load("data/thrml_spw/positive_mnist/seed0.npz")
    theta, biases, _, _, _ = initial_arrays(
        Path("data/thrml_spw/positive_mnist/seed0.npz"))
    x = data["validation_inputs"][:8]
    samples = 16
    layer0 = static_graph(N0)
    layer1 = static_graph(N1)
    key0, key1 = jax.random.split(jax.random.key(44001))

    observed_pre0 = np.asarray(native_sampled_output(
        theta[0], key0, jnp.asarray(x), biases[0], W0_SHAPE, *layer0, samples))
    matrices0 = explicit_layer_samples(theta[0], key0, *layer0, samples).reshape(
        (samples, *W0_SHAPE))
    explicit_pre0 = np.einsum("bi,tio->tbo", x, matrices0).mean(axis=0) + np.asarray(biases[0])
    err0 = float(np.max(np.abs(observed_pre0 - explicit_pre0)))
    hidden = np.maximum(observed_pre0, 0.0)

    observed_logits = np.asarray(native_sampled_output(
        theta[1], key1, jnp.asarray(hidden), biases[1], W1_SHAPE, *layer1, samples))
    matrices1 = explicit_layer_samples(theta[1], key1, *layer1, samples).reshape(
        (samples, *W1_SHAPE))
    explicit_logits = np.einsum("bi,tio->tbo", hidden, matrices1).mean(axis=0) + np.asarray(biases[1])
    err1 = float(np.max(np.abs(observed_logits - explicit_logits)))

    # Deliberately wrong schedule: activate each noisy first-layer sample before
    # averaging, then pair it with layer-1 samples. This must differ from ReLU(mean).
    sampled_pre0 = np.einsum("bi,tio->tbo", x, matrices0) + np.asarray(biases[0])[None,None,:]
    hidden_early = np.maximum(sampled_pre0, 0.0)
    early_logits = np.einsum("tbi,tio->tbo", hidden_early, matrices1).mean(axis=0) + np.asarray(biases[1])
    early_difference = float(np.max(np.abs(early_logits - explicit_logits)))
    result = {"samples": samples, "observer_vs_sample_states_layer0_max_abs": err0,
              "observer_vs_sample_states_layer1_max_abs": err1,
              "incorrect_relu_before_averaging_max_abs_difference": early_difference,
              "sequential_identity_tolerance": 2e-5,
              "passed_observer_identity": err0 < 2e-5 and err1 < 2e-5,
              "passed_invalid_schedule_detector": early_difference > 1e-4}
    if not result["passed_observer_identity"] or not result["passed_invalid_schedule_detector"]:
        raise RuntimeError(f"native schedule self-test failed: {result}")
    return result


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,nargs="+",default=[0])
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--native-samples",type=int,default=16)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--learning-rate",type=float,default=1e-3)
    parser.add_argument("--data",default="data/thrml_spw/native_training/training_data.npz")
    parser.add_argument("--out",default="results/thrml_spw/native_training")
    parser.add_argument("--test-examples",type=int,default=1024)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.native_samples not in (16,64):
        raise ValueError("native-samples must be 16 or 64")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    schedule_test = native_schedule_self_test()
    print({"native_schedule_test": schedule_test}, flush=True)
    if args.self_test:
        (out/"schedule_self_test.json").write_text(json.dumps(schedule_test,indent=2)+"\n")
        return
    data=np.load(args.data)
    train_x,train_y=data["train_inputs"],data["train_labels"]
    val_x,val_y=data["validation_inputs"],data["validation_labels"]
    # Paired held-out test set is reserved from method selection; only evaluate/export
    # final outputs after all training decisions are fixed.
    chosen=np.sort(np.random.default_rng(20261027).choice(len(data["test_labels"]),
                       min(args.test_examples,len(data["test_labels"])),replace=False))
    test=(data["test_inputs"][chosen],data["test_labels"][chosen])
    layer0=static_graph(N0)
    layer1=static_graph(N1)
    rows=[];all_results={};start=time.monotonic()
    for seed in args.seeds:
        npz_path=Path("data/thrml_spw/positive_mnist")/f"seed{seed}.npz"
        for kind in ("expected_control","native_thrml"):
            result=train_condition(kind,seed,args.epochs,args.native_samples,
                args.batch_size,args.learning_rate,(train_x,train_y,val_x,val_y),
                layer0,layer1)
            name=f"seed{seed}_{kind}"
            p,b=export_weights(out/f"{name}_weights.npz",result["theta"],result["biases"])
            initial_theta,initial_b,_,_,_=initial_arrays(npz_path)
            before_expected=validation_metrics(initial_theta,initial_b,val_x,val_y,args.batch_size)
            before_native=native_validation(initial_theta,initial_b,val_x,val_y,seed+20000,
                args.native_samples,layer0,layer1,args.batch_size)
            after_expected=validation_metrics(result["theta"],result["biases"],val_x,val_y,args.batch_size)
            after_native=native_validation(result["theta"],result["biases"],val_x,val_y,
                seed+21000,args.native_samples,layer0,layer1,args.batch_size)
            test_result=evaluate_test_native(result["theta"],result["biases"],test,
                seed+22000,args.native_samples,layer0,layer1,args.batch_size,out,name)
            row={"seed":seed,"method":kind,"epochs":args.epochs,
                 "native_samples_training":args.native_samples if kind=="native_thrml" else 0,
                 "train_loss_last":result["losses"][-1],
                 "validation_expected_before_accuracy":before_expected["accuracy"],
                 "validation_expected_after_accuracy":after_expected["accuracy"],
                 "validation_native_before_accuracy":before_native["accuracy"],
                 "validation_native_after_accuracy":after_native["accuracy"],
                 "validation_expected_after_ce":after_expected["cross_entropy"],
                 "validation_native_after_ce":after_native["cross_entropy"],
                 "test_native_accuracy":test_result["test_accuracy"],
                 "test_native_ce":test_result["test_cross_entropy"],
                 "test_examples":len(test[1]),"weights_clipped_initially":int(sum(np.count_nonzero((w<LOWER_P)|(w>UPPER_P)) for w in result["original_weights"])),
                 "all_exported_weights_in_open_unit_interval":bool(all(np.all((q>0)&(q<1)) for q in p)),
                 "sampled_nodes":N_WEIGHT,"gibbs_updates_training_estimate":int(args.epochs*len(prepare_batches(train_x,train_y,seed,args.batch_size)[0])*args.native_samples*N_WEIGHT) if kind=="native_thrml" else 0}
            rows.append(row)
            all_results[name]={"epoch_metrics":result["epoch_metrics"],"row":row,
                "source_checkpoint":str(npz_path),"checkpoint_sha256":sha256(npz_path),
                "batch_order_sha256":hashlib.sha256(result["batch_order"].astype(np.int64).tobytes()).hexdigest(),
                "export_sha256":sha256(out/f"{name}_weights.npz")}
            print(row,flush=True)
    with (out/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    details={"config":vars(args),"fixed_data_path":args.data,
        "dataset_sizes":{"train":len(train_y),"validation":len(val_y),"test_paired":len(test[1])},
        "architecture":"64->32->10 ReLU, w in (0,1), deterministic biases",
        "static_graph":{"free_pbit_nodes_by_layer":{"layer0":N0,"layer1":N1},"clamped_nodes":0,
            "node_update":"two static free blocks/programs, one per layer; all weight spins in that layer are updated each Gibbs step; independent layer keys",
            "field_mapping":"p=sigmoid(theta), unary Ising field theta/2 at beta=1",
            "sampler":"official THRML FactorSamplingProgram / SpinEBMFactor / SpinGibbsConditional / sample_with_observation"},
        "initialization":"pretrained weights clipped to [1e-4,1-1e-4] before logit; number changed is recorded",
        "gradient":"biased STE: native THRML sampled preactivation forward; stop-gradient correction; derivative through expected linear weights only. No gradient through Gibbs sampling, no claim of unbiased gradient.",
        "controls":"expected-weight fine-tuning and native THRML T-sample fine-tuning use same seed-specific model, fixed precomputed minibatch order, Adam and learning rate",
        "validation_policy":"fixed 5,000-row validation set; no test metrics used for selection",
        "native_schedule_test":schedule_test,
        "paired_test_indices":chosen.tolist(),"runs":all_results,
        "data_sha256":sha256(args.data),"script_sha256":sha256(Path(__file__)),
        "elapsed_seconds":time.monotonic()-start}
    (out/"details.json").write_text(json.dumps(details,indent=2)+"\n")
    manifest(out/"manifest.json",vars(args),{"rows":rows,"elapsed_seconds":details["elapsed_seconds"],
        "script_sha256":details["script_sha256"],"data_sha256":details["data_sha256"],
        "limitations":details["gradient"]})
    print(f"saved native fine-tuning artifacts in {out}",flush=True)


if __name__=="__main__":main()
