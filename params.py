import hashlib
import json
import os

class Parameters:
    n_datasize: int
    n_blocks: int
    model_name: str
    dataset_name: str
    # TODO: n_toks cutoff
    string_size_cutoff: int
    n_datasize: int
    metric_width: float
    seed: int

    def get_json(self):
        return {
            'seed': self.seed,
            'n_datasize': self.n_datasize,
            'n_blocks': self.n_blocks,
            'string_size_cutoff': self.string_size_cutoff,
            'model_name': self.model_name,
            "metric_width": self.metric_width,
            'dataset_name': self.dataset_name
        }

    def create_param_tag(self):
        m = hashlib.sha256()
        m.update(json.dumps(self.get_json()).encode('utf-8'))
        return m.hexdigest()[:40]
	
    def load_from_json(self, json):
        self.seed = json['seed']
        self.n_datasize = json['n_datasize']
        self.n_blocks = json['n_blocks']
        self.string_size_cutoff = json['string_size_cutoff']
        self.model_name = json['model_name']
        self.metric_width = json['metric_width']
        self.dataset_name = json['dataset_name']


    def get_and_prepare_save_tag(self, prepend: str):
        if not os.path.exists('metadata'):
            os.mkdir('metadata')
        if not os.path.exists(f'metadata/{self.create_param_tag()}'):
            os.mkdir(f'metadata/{self.create_param_tag()}')
            json.dump(self.metadata_json(), open(
                f'metadata/{self.create_param_tag()}/metadata.json', 'w'))
        return f'metadata/{self.create_param_tag()}/{prepend}.pkl'

