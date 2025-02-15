import pandas as pd
import numpy as np

class SampleAllocator:
    def __init__(self, expected_uas: dict[str, float], target_error: float):
        self.expected_uas = expected_uas
        self.target_error = target_error

    def allocate(self):
        pass

class ProportionalAllocator(SampleAllocator):
    def __init__(self, expected_uas: dict[str, float], target_error: float):
        super().__init__(expected_uas, target_error)

    def allocate(self, pixel_counts: dict[str, int]):
        print('Using Proportional Allocator to allocate number of samples to strata.')
        #TODO deal with zero counts, currenlty they are not being considered
        
        total_pixels = sum(pixel_counts.values())
        df = pd.DataFrame.from_dict(pixel_counts, orient='index', columns=['count'])
        df['ua'] = df.index.map(self.expected_uas)
        df['wh'] = df['count'] / total_pixels
        df['s'] = np.sqrt(df['ua'] * (1 - df['ua']))
        df['nInt'] = df['wh'] * df['s']

        n = round(np.square(df['nInt'].sum() / self.target_error))
        df['nh'] = round(df['wh'] * n)
        df['sampling_rate'] = df['nh'] / df['count']

        print(df)
        
        return df['nh'].to_dict()
    
    def compute_expected_error(self, sample_assignment: dict[str, int]):
        pass


class NeymanAllocator(SampleAllocator):
    def __init__(self, expected_uas: dict[str, float], target_error: float):
        super().__init__(expected_uas, target_error)

    def allocate(self, pixel_counts: dict[str, int]):
        pass

    def compute_expected_error(self, sample_assignment: dict[str, int]):
        pass