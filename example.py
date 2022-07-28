"""
MSF-GC. This module provides a basic demonstration
Copyright (C) 2022  Raymond Leung

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Further information about this program can be obtained from:
- Raymond Leung (raymond.leung@sydney.edu.au)
"""

import numpy as np
import pandas as pd
import shapely.wkt
from shapely.geometry import Polygon
from algorithm_implementation import GradeBlockReliabilityAssessment


def compute(algorithm, gb_id, gb_polygon, gb_holes_xy, gb_chemistry, randseeds=[8365]):
    #initialise result object
    results = {
       'grade_block_id': gb_id,
       'score_spatial_confidence': np.nan,
       'score_chemical_consensus': np.nan,
       'score_overall_reliability': np.nan,
       'factor_gradeblock_coverage': np.nan,
       'factor_hole_entropies': np.nan,
       'factor_sampling_density': np.nan,
       'stats_hole_sparsity': np.nan,
       'stats_rd_gmean': np.nan,
       'stats_rd_score': np.nan,
       'stats_eta_outliers': np.nan,
       'stats_area': np.nan,
       'stats_hole_count': np.nan,
       'stats_effective_samples': np.nan
       }
    return algorithm.run(gb_polygon, gb_holes_xy, gb_chemistry, results, randseeds)


if __name__ == "__main__":
    #read data into dictionaries
    df_bh = pd.read_csv('data/blasthole_assays.csv')
    df_gb = pd.read_csv('data/gradeblock_geometries.csv')
    uniq_gradeblock_ids = df_gb['gradeblock_id'].values

    #define lambda functions to access information from dataframes
    retrieve = lambda df, cols: [df[df['gradeblock_id']==g][cols].values
                                 for g in uniq_gradeblock_ids]
    holes_properties = lambda cols: retrieve(df_bh, cols)
    block_properties = lambda cols: retrieve(df_gb, cols)
    gb_names = dict(zip(uniq_gradeblock_ids, block_properties('name')))

    #specify the required input
    gb_boundaries = dict(zip(df_gb['gradeblock_id'].values,
                        [shapely.wkt.loads(wkt) for wkt in df_gb['boundary'].values]))
    blasthole_xy = dict(zip(uniq_gradeblock_ids, holes_properties(['x','y'])))
    blasthole_assays = dict(zip(uniq_gradeblock_ids, holes_properties(['c1','c2','c3'])))
    config = {'modelled_elements': ['Fe','SiO2','Al2O3'],
              'chemical_weights': [0.5, 0.325, 0.175]}

    #instantiate object
    algorithm = GradeBlockReliabilityAssessment(config)
    outcomes = {}

    #analyse individual grade-blocks
    for i in uniq_gradeblock_ids:
        gb_polygon = gb_boundaries[i]
        gb_holes_xy = blasthole_xy[i]
        gb_chemistry = blasthole_assays[i]
        outcomes[i] = compute(algorithm, i, gb_polygon, gb_holes_xy, gb_chemistry)
        print(f'grade-block {gb_names[i]}:\n{outcomes[i]}\n')

    #split-sequence analysis for eta=0.4375
    chemistry = np.genfromtxt('data/high_breakdown.csv', skip_header=1, delimiter=',')
    output1 = algorithm.compute_geochemical_consensus(chemistry, handle_breakdown=True)
    #- Alternative: if used within the context of algorithm.run(),
    #  config['handle_breakdown'] should be set to True before class
    #  object initialisation to enable split-sequence analysis.
    row_fmt = lambda x, y: '\n'.join([str(i) for i in list(zip(x,y[:-1]))])
    description = ['eta_outliers', 'rd_gmean', 'rd_score', 'f_consensus']
    output2 = algorithm.compute_geochemical_consensus(chemistry, handle_breakdown=False)
    print('split-sequence analysis test case:\n- enabled:\n{}\n- disabled:\n{}\n'.format(
           row_fmt(description, output1), row_fmt(description, output2)))
