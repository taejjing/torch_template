import argparse
import os
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from pathlib import Path


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        eval = config['data_loader']['args']['eval'],
        num_workers=config['data_loader']['args']['num_workers']
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

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

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    result = []
    with torch.no_grad():
        for i, (data, target, file_path) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # save sample images, or do something with output here
            label = target.cpu().numpy().astype(int)
            predict = (output.cpu().numpy() >= 0.5).astype(int)
            score = output.cpu().numpy()

            batch = list(zip(file_path, label, predict, score))
            result.append(batch)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                # mid = metric(output, target) * batch_size
                total_metrics[i] += metric(output, target) * batch_size
                    
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    with open(result_path.joinpath('eval_result.csv'), 'w') as writer:
        writer.write(f"path,target,predict,score\n")
        for batch in result:
            writer.writelines(map(lambda x: f"{x[0]},{x[1]},{x[2]},{x[3]}\n", batch))
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
