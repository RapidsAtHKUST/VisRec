import functools

from utils import dist_util, logger
from utils.train_utils.base_train_util import TrainLoop

from utils.data_utils.transform_util import *
from utils.data_utils.image_util import magnitude, tile_image

uv_dense1 = np.load("data/uv_dense.npy") 
uv_dense1 = torch.tensor(uv_dense1).to(dist_util.dev())

class UNetTrainLoop(TrainLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_process(self, batch):
        ksapce_c, args_dict, N_s, N_u = batch
        vis_sparse_zf = args_dict["vis_sparse_zf"]
        vis_dense = args_dict["vis_dense"]
        return vis_sparse_zf, vis_dense, N_s, N_u

    def img_loss(self, output, label):
        return (to_img_th(output[:,0,...], output[:,1,...], uv_dense1) - to_img_th(label[:,0,...], label[:,1,...], uv_dense1)).abs().mean()

    def forward_backward(self, batch):
        batch, label, N_s, N_u = self.batch_process(batch)
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
 
            micro_input = torch.from_numpy(batch).to(dist_util.dev())
            micro_label = label.to(dist_util.dev())

            last_batch = (i + self.microbatch) >= batch.shape[0]
            micro_output = self.ddp_model(micro_input)

            micro_label_s = micro_label[:N_s]
            micro_output_s = micro_output[:N_s]

            micro_label_u = micro_output[-N_u:]
            micro_output_u = micro_output[N_s:N_s+N_u]

            compute_loss_s = functools.partial(
                th.nn.functional.mse_loss,
                micro_output_s,
                micro_label_s
            )

            compute_loss_u = functools.partial(
                th.nn.functional.mse_loss,
                micro_output_u,
                micro_label_u
            )


            if last_batch or not self.use_ddp:
                print(compute_loss_s(), compute_loss_u())
                print(self.img_loss(micro_label_s, micro_output_s), self.img_loss(micro_label_u, micro_output_u))
                loss = compute_loss_s() + 0.1 * compute_loss_u() 
            else:
                with self.ddp_model.no_sync():
                    print(compute_loss_s(), compute_loss_u())
                    loss = compute_loss_s() + 0.1 * compute_loss_u() 

            logger.log_kv("loss", loss)
            self.mp_trainer.backward(loss)

            self._post_process(micro_input, micro_label, micro_output, i)

    def _post_process(self, micro_input, micro_label, micro_output, i):
        if self.step % self.save_interval == 0 and i == 0:
            ncols = len(micro_input)

