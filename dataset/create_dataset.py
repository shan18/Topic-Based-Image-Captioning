import os
import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument(
        '--root', default=os.path.dirname(os.path.abspath(__file__)),
        help='Root directory containing the dataset folders'
    )
    parser.add_argument(
        '--raw', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument('--categories', action='store_true', help='Encode image categories')
    parser.add_argument('--captions', action='store_true', help='Encode image captions')
    parser.add_argument('--train_split', default=0.8, help='Training data split')
    parser.add_argument('--val_split', default=0.19, help='Validation data split')
    args = parser.parse_args()

    if args.train_split <= 0 or args.train_split >= 1:
        print('Value of train_split should be between 0 and 1')
    elif args.val_split <= 0 or args.val_split >= 1:
        print('Value of val_split should be between 0 and 1')
    elif args.train_split <= args.val_split:
        print('train_split should be greater than val_split')
    elif args.train_split + args.val_split >= 1:
        print('Please enter a valid split')
    else:
        main(args)
