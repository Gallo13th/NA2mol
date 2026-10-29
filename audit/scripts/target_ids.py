"""Read structure identifiers as strings and reject spreadsheet conversions."""
import re

KNOWN_ALIASES = {'1.00e+66': '1e66', '3.00e+37': '3e37'}

def structure_id(value, repair_known=False):
    value = str(value).strip().lower()
    if value in KNOWN_ALIASES and repair_known:
        return KNOWN_ALIASES[value]
    if not re.fullmatch(r'[0-9][a-z0-9]{3}', value):
        raise ValueError(f'Invalid structure identifier: {value!r}; preserve IDs as strings at ingestion')
    return value

def validate_join_keys(reference, overlap, generated):
    for frame in (reference, overlap, generated):
        frame['target_id'].map(structure_id)
    keys = ['benchmark', 'target_id', 'canonical']
    r = reference[keys].drop_duplicates()
    o = overlap[keys].drop_duplicates()
    joined = r.merge(o, on=keys, how='left', indicator=True, validate='one_to_one')
    if not (joined['_merge'] == 'both').all():
        raise ValueError('Reference-to-overlap join has unmatched keys')
    return True
