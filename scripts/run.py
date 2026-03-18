import argparse
from src.processor import IrisProcessor
import os
from tqdm import tqdm
import multiprocessing as mp


def process_pair(args):
    img1_path, img2_path, dataset_name, thresh = args

    processor = IrisProcessor(dataset_name)
    score = processor.compute_score(img1_path, img2_path)

    label = 1 if score <= thresh else 0
    return img1_path, img2_path, label


def main():
    thresh = 0.4

    parser = argparse.ArgumentParser(description='Ganzin Iris Recognition Challenge')
    parser.add_argument('--input', type=str, metavar='PATH', required=True,
                        help='Input file to specify a list of sampled pairs')
    parser.add_argument('--output', type=str, metavar='PATH', required=True,
                        help='A list of sampled pairs and the testing results')

    args = parser.parse_args()

    name_no_ext, _ = os.path.splitext(args.input)
    parts = name_no_ext.split("-")
    dataset_name = parts[-1]

    with open(args.input, 'r') as in_file, open(args.output, 'w') as out_file:
        lines = [line.strip() for line in in_file if line.strip()]
    
    tasks = []
    for line in lines:
        img1_path, img2_path = [p.strip() for p in line.split(",")]
        tasks.append((img1_path, img2_path, dataset_name, thresh))

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_pair, tasks), total=len(tasks)))
    
    with open(args.output, "w") as out_file:
        for img1, img2, label in results:
            out_file.write(f"{img1}, {img2}, {label}\n")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()