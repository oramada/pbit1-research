"""Finite-control native probability training with an explicit biased STE."""
import jax
import jax.numpy as jnp
import numpy as np
from .native_training import static_graph,native_sampled_output

METHODS=('expected','native_ideal','native_device')


def command(theta):
    code=np.rint((np.clip(np.asarray(theta,dtype=float)/2,-4,4)+4)*255/8).astype(np.uint8)
    return (-4+code.astype(float)*8/255).astype(np.float32)


def initial_theta(weights):
    q=np.concatenate([w.reshape(-1) for w in weights]).astype(float)
    eps=1/(1+np.exp(8));q=np.clip(q,eps,1-eps)
    return (np.log(q)-np.log1p(-q)).astype(np.float32)


def profile(seed,index,domain):
    # Domain-specific seeds keep training, validation and test devices disjoint.
    bases={'train':2026110600,'validation':2026111600,'test':2026112600}
    if domain not in bases:raise ValueError(domain)
    z=np.random.default_rng(np.random.SeedSequence([bases[domain],seed,index])).standard_normal((2,2368))
    return np.exp(.2*z[1]-.02),.2*z[0]


class TrainingNetwork:
    def __init__(self,biases,samples=64):
        self.bias=[jnp.asarray(b) for b in biases];self.graph=[static_graph(2048),static_graph(320)];self.samples=samples

    def objective(self,method):
        if method not in METHODS:raise ValueError(method)
        def loss(theta,inputs,labels,key,fields):
            p=jax.nn.sigmoid(theta);p0=p[:2048].reshape(64,32);p1=p[2048:].reshape(32,10)
            mean0=inputs@p0+self.bias[0]
            if method=='expected':
                pre=mean0;hidden=jax.nn.relu(pre);logits=hidden@p1+self.bias[1]
            else:
                # The caller emits finite F32 fields from F64 control/device
                # arithmetic, held fixed across this minibatch's native draws.
                k0,k1=jax.random.split(key)
                n0=native_sampled_output(2*fields[:2048],k0,inputs,self.bias[0],(64,32),*self.graph[0],self.samples)
                pre=mean0+jax.lax.stop_gradient(n0-mean0)
                hidden=jax.nn.relu(pre);mean1=hidden@p1+self.bias[1]
                n1=native_sampled_output(2*fields[2048:],k1,hidden,self.bias[1],(32,10),*self.graph[1],self.samples)
                logits=mean1+jax.lax.stop_gradient(n1-mean1)
            ce=-jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(labels.size),labels])
            return ce,(pre,logits)
        return jax.jit(jax.value_and_grad(loss,has_aux=True))


def adam(theta,grad,m,v,step,learning_rate=.001):
    m=.9*m+.1*grad;v=.999*v+.001*grad*grad
    theta=jnp.clip(theta-learning_rate*(m/(1-.9**step))/(jnp.sqrt(v/(1-.999**step))+1e-8),-8,8)
    return theta,m,v


def manual_ste_gradient(theta,x,y,pre,logits):
    # Explicit expected-linear backward rule, not the gradient of the discrete
    # Bernoulli likelihood or an unbiased estimator of noisy task gradients.
    p=1/(1+np.exp(-theta.astype(float)));p1=p[2048:].reshape(32,10)
    z=logits.astype(float)-logits.max(axis=1,keepdims=True);q=np.exp(z);q/=q.sum(axis=1,keepdims=True)
    ce=-np.mean(np.log(q[np.arange(len(y)),y]));q[np.arange(len(y)),y]-=1;q/=len(y)
    hidden=np.maximum(pre.astype(float),0)
    g1=(hidden.T@q).reshape(-1)*p[2048:]*(1-p[2048:])
    back=(q@p1.T)*(pre>0)
    g0=(x.astype(float).T@back).reshape(-1)*p[:2048]*(1-p[:2048])
    return ce,np.concatenate([g0,g1])
