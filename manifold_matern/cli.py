import argparse

def argument_parser():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num-eigenpairs', type=int, default=500,
                           help='Number of eigenpairs to use. Default is 500')
    argparser.add_argument('--seed', type=int,
                           help='Random seed')
    argparser.add_argument('--output-dir', type=str, default='output',
                           help='Output directory to save .pvd files to. Default is ./output')
    argparser.add_argument('--eigenpairs-file', type=str,
                           help='.npy file with precomputed eigenpairs')
    argparser.add_argument('--mayavi', action='store_true',
                           help='Render results to .png with Mayavi')
    argparser.add_argument('--num-samples', type=int, default=16,
                           help='Number of random samples to generate')

    return argparser
