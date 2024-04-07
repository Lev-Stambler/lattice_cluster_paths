import hashlib
import json
import os

class LatticeParams:
    # Set to -1 to start at the last layer
    top_layer_idx = -1
    max_n_parents: int

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
    

class InterpParams:
    n_datasize: int
    n_blocks: int
    model_name: str
    model_n_dims: int
    dataset_name: str
    # TODO: n_toks cutoff
    string_size_cutoff: int
    n_datasize: int
    seed: int

    lattice: LatticeParams

    def __init__(self, lattice_params: LatticeParams, **kwargs) -> None:
        self.lattice = lattice_params
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_json_data(self):
        return {
            'seed': self.seed,
            'n_datasize': self.n_datasize,
            'n_blocks': self.n_blocks,
            'string_size_cutoff': self.string_size_cutoff,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name
        }

    def get_json_correlation(self):
        return {
            'dataset_info': self.get_json_data()
        }
    
    def get_json_lattice(self):
        return {
            'correlation_and_data': self.get_json_correlation(),
            'lattice_params': LatticeParams.__dict__
        }

    def _create_param_tag_data(self):
        m = hashlib.sha256()
        m.update(json.dumps(self.get_json_data()).encode('utf-8'))
        return m.hexdigest()[:40]

    def _create_param_tag_correlation(self):
        m = hashlib.sha256()
        m.update(json.dumps(self.get_json_correlation()).encode('utf-8'))
        return m.hexdigest()[:40]
    
    def load_from_json(self, json):
        raise NotImplementedError
        self.seed = json['seed']
        self.n_datasize = json['n_datasize']
        self.n_blocks = json['n_blocks']
        self.string_size_cutoff = json['string_size_cutoff']
        self.model_name = json['model_name']
        self.metric_width = json['metric_width']
        self.dataset_name = json['dataset_name']

    def get_and_prepare_data_save_tag(self, prepend: str):
        """
        Associated tag for specifically just the data. 
        """
        if not os.path.exists('metadata'):
            os.mkdir('metadata')
        if not os.path.exists(f'metadata/data-{self._create_param_tag_data()}'):
            os.mkdir(f'metadata/data-{self._create_param_tag_data()}')
            json.dump(self.get_json_data(), open(
                f'metadata/data-{self._create_param_tag_data()}/metadata.json', 'w'))
        return f'metadata/data-{self._create_param_tag_data()}/{prepend}.pkl'

    def get_and_prepare_correlation_save_tag(self, prepend: str):
        if not os.path.exists('metadata'):
            os.mkdir('metadata')
        if not os.path.exists(f'metadata/correlation-{self._create_param_tag_correlation()}'):
            os.mkdir(f'metadata/correlation-{self._create_param_tag_correlation()}')
            json.dump(self.get_json_correlation(), open(
                f'metadata/correlation-{self._create_param_tag_correlation()}/metadata.json', 'w'))
        return f'metadata/correlation-{self._create_param_tag_correlation()}/{prepend}.pkl'
