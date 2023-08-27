from lib.datasets.dataset_factory import dataset_factory, _sample_factory

class DerivedDataset:
    def __init__(self, base_class, task_class, *args, **kwargs):
        self.base_class_instance = base_class(*args, **kwargs)
        self.task_class_instance = task_class()

    def __getattr__(self, attr):
        try:
            return getattr(self.base_class_instance, attr)
        except AttributeError:
            return getattr(self.task_class_instance, attr)

    def __len__(self):
        return len(self.base_class_instance)

    def __getitem__(self, index):
        return self.base_class_instance[index]

class CustomDataset:
    def __init__(self, dataset, task):
        self.base_class = dataset_factory[dataset]
        self.task_class = _sample_factory[task]

    def __call__(self, *args, **kwargs):
        return DerivedDataset(self.base_class, self.task_class, *args, **kwargs)