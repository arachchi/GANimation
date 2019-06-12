from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('-in', '--input_path', type=str, help='path to image')
        self._parser.add_argument('-out', '--output_dir', type=str, default='./output', help='output path')
        self.is_train = False
