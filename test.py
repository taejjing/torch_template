import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from pathlib import Path


def main(config: ConfigParser):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        data_dir=config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=config['data_loader']['args']['num_workers']
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # test result 
    result_path = Path(config['result_dir']).joinpath(config['name'])
    os.makedirs(result_path, exist_ok=True)
    
    result = []
    with torch.no_grad():
        for i, (data, target, file_path) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.sigmoid(output)

            # save sample images, or do something with output here
            predict = (output.cpu().numpy() >= 0.5).astype(int)
            score = output.cpu().numpy()

            batch = list(zip(file_path, predict, score))
            result.append(batch)

    
    with open(result_path.joinpath('test_result.csv'), 'w') as writer:
        writer.write(f"path,predict,score\n")
        for batch in result:
            writer.writelines(map(lambda x: f"{x[0]},{x[1]},{x[2]}\n", batch))
        logger.info("Done")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
