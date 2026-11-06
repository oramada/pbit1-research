"""Metadata-only complete holdout batching and loss assembly.

This component neither opens token arrays nor authorizes model inference. A
future frozen scoring protocol must bind its batch plan and audit actual input
IDs, raw logits, and loss calculations independently.
"""
import numpy as np


def batch_plan(manifest):
    if manifest.get('status') != 'article_separated_holdout_frozen_no_model_scores':
        raise ValueError('Unrecognized reserved-data manifest')
    if manifest.get('total_predicted_targets') != 10240:
        raise ValueError('The complete reserved evaluation is required')
    cohorts = manifest['cohorts']
    plans, seen_articles = [], set()
    global_target = 0
    for label, windows, length, repeats, batch_size in [
            ('short', 96, 65, 4, 8), ('long', 16, 257, 1, 4)]:
        cohort = cohorts[label]
        ids, offsets = cohort['article_ids'], cohort['selected_offsets']
        if (cohort['windows'], cohort['window_length'], cohort['predicted_tokens']) != (
                windows, length, windows * (length - 1)):
            raise ValueError('Reserved cohort geometry changed')
        if len(ids) != windows // repeats or len(set(ids)) != len(ids):
            raise ValueError('Missing or repeated article clusters')
        if seen_articles.intersection(ids) or len(offsets) != windows:
            raise ValueError('Cross-stratum article overlap or missing windows')
        seen_articles.update(ids)
        if any(type(v) is not int or v < 0 for v in ids + offsets):
            raise ValueError('Article IDs and offsets must be nonnegative integers')
        article_records = {a['article_id']: a for a in manifest['articles']}
        expanded = [article for article in ids for _ in range(repeats)]
        previous_end = {}
        for article, offset in zip(expanded, offsets, strict=True):
            record = article_records[article]
            if not record['token_start'] <= offset < offset + length <= record['token_end']:
                raise ValueError('Window extends outside its assigned article')
            if offset < previous_end.get(article, -1):
                raise ValueError('Article windows overlap or are out of order')
            previous_end[article] = offset + length
        for start in range(0, windows, batch_size):
            stop = min(start + batch_size, windows)
            count = (stop - start) * (length - 1)
            plans.append({'batch_index': len(plans), 'cohort': label,
                'window_start': start, 'window_stop': stop,
                'input_shape': [stop - start, length],
                'loss_shape': [stop - start, length - 1],
                'article_ids_by_window': expanded[start:stop],
                'source_token_offsets': offsets[start:stop],
                'global_target_start': global_target,
                'global_target_stop': global_target + count})
            global_target += count
    if global_target != 10240 or len(seen_articles) != 40:
        raise ValueError('Incomplete evaluation')
    return plans


def assemble_losses(plan, records):
    """Assemble one seed/stage's complete losses; reject omissions/duplicates.

The caller supplies a manifest-derived plan. Records may arrive out of order,
but each must have exactly batch_index and token_NLL. This is arithmetic
assembly, not proof of provenance or recomputation from raw logits.
"""
    if len(plan) != 16 or [p['batch_index'] for p in plan] != list(range(16)):
        raise ValueError('Expected the complete sixteen-batch plan')
    arrays = {'short': np.empty((96, 64)), 'long': np.empty((16, 256))}
    coverage = {name: np.zeros(a.shape, dtype=np.uint8) for name, a in arrays.items()}
    seen = set()
    for record in records:
        if set(record) != {'batch_index', 'token_NLL'}:
            raise ValueError('Unexpected loss record schema')
        index = record['batch_index']
        if type(index) is not int or not 0 <= index < len(plan) or index in seen:
            raise ValueError('Invalid or duplicate batch')
        seen.add(index)
        spec = plan[index]
        values = np.asarray(record['token_NLL'], dtype=np.float64)
        if list(values.shape) != spec['loss_shape'] or not np.isfinite(values).all():
            raise ValueError('Incorrect loss shape or nonfinite values')
        name, start, stop = spec['cohort'], spec['window_start'], spec['window_stop']
        arrays[name][start:stop] = values
        coverage[name][start:stop] += 1
    if len(seen) != len(plan) or any(not np.all(v == 1) for v in coverage.values()):
        raise ValueError('Every target must occur exactly once')
    return arrays['short'], arrays['long']
