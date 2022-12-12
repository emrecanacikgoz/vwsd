import os.path as osp


def process_path(path):
    return osp.abspath(osp.expanduser(path))


def write_results(results, path):
    path = process_path(path)
    image_files = []
    for batch in results:
        image_files.extend(batch['image_files'])
    with open(path, 'w') as f:
        for this in image_files:
            line = ' '.join(this) + '\n'
            f.write(line)