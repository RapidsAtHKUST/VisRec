import matplotlib.pyplot as plt

from utils.test_utils.base_test_util import *



class UNetTestLoop(TestLoop):

    def __init__(self, microbatch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.microbatch = microbatch
        assert microbatch >= 1
        assert self.batch_size == 1

        self.curr_file = {
            "file_name": "",
            "vis_dense": [],
            "vis_sparse_zf": [],
            "img_d": [],
            "mask": [],
            "scale_coeff": [],
            "slice_index": [],
        }


    def run_loop(self):
        super().run_loop()
        self.reconstruct_for_one_volume()


    def reconstruct_for_one_volume(self):
        for key in ["vis_dense", "vis_sparse_zf", "img_d", "mask"]:
            self.curr_file[key] = th.cat(self.curr_file[key], dim=0).to(dist_util.dev())
        outputs = []
        for i in range(0, len(self.curr_file["slice_index"]), self.microbatch):
            micro_input = self.curr_file["vis_sparse_zf"][i: i + self.microbatch]
            with th.no_grad():
                micro_output = self.model(micro_input)
            outputs.append(micro_output)
        outputs = th.cat(outputs, dim=0)


       

    def forward_backward(self, data_item):
        ksapce_c, batch_args = data_item

        if self.curr_file["file_name"] != batch_args["file_name"] and self.curr_file["file_name"] != "":
            self.reconstruct_for_one_volume()

        if self.curr_file["file_name"] != batch_args["file_name"]:
            for key in self.curr_file.keys():
                self.curr_file[key] = []
            self.curr_file["file_name"] = batch_args["file_name"]

        for key in self.curr_file.keys():
            if key == "file_name":
                continue
            else:
                self.curr_file[key].append(batch_args[key])
