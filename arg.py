import argparse
import yaml

def get_parser():
    parser = argparse.ArgumentParser()
    
    #For config file path
    parser.add_argument('--config', type=str, default=None, 
                        help='Configuration file path (YAML)')
    
    #For experiment settings:
    parser.add_argument('--log',type=bool,default=True,help="choose if log")
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer','Chameleon','Squirrel','Actor','PubMed','Photo','Computers'])
    parser.add_argument('--imbtype', type=str, default='classnum', choices=['classnum', 'topology'])
    parser.add_argument('--imb_ratio', type=float, default=20)
    parser.add_argument('--shuffle_seed', type=int, default=25, help="shuffle_seed")
    parser.add_argument('--random_seed', type=int, default=42, help="random_seed")
    parser.add_argument('--backbone', type=str, default='GCN', choices=['GCN', 'Sage'])
    parser.add_argument('--metric', type=str, default='loss', choices=['acc','loss'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--least_epoch', type=int, default=50)
    parser.add_argument('--early_stop', type=int, default=10)
    
    #For backbone settings:
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--adj', type=str, default='norm')
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    
    #For Pre-Train settings:
    parser.add_argument('--least_Pre_epochs', type=int, default=40)
    parser.add_argument('--pre_epochs', type=int, default=500)
    parser.add_argument('--pre_lr', type=float, default=0.001)

    #For graph structure refinement settings:
    parser.add_argument('--graph_skip_conn', type=float, default=0.8)
    parser.add_argument('--update_adj_ratio', type=float, default=0.0)
    parser.add_argument('--iters', type=int, default=0)
    parser.add_argument('--iters2', type=int, default=5)
    
    #For mixup settings:
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--mixnum', type=float, default=0.3)
    parser.add_argument('--lam', type=float, default=0.5)
    #For prompt settings:
    parser.add_argument('--pmt_lr', type=float, default=0.005)
    parser.add_argument('--pmt_weight_decay', type=float, default=0.0005)
    parser.add_argument('--pmt_dropout', type=float, default=0)
 
    
    return parser

def update_params(parser: argparse.ArgumentParser):
    args, _ = parser.parse_known_args() 
    config_path = args.config
    
    if config_path is not None:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            parser.set_defaults(**{key: value})
    
    final_args = parser.parse_args()
    return final_args