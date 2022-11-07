"""
MSF-GC. This module implements the algorithms described in doi.org/10.31224/2422
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
import sys
import warnings
from scipy.spatial import cKDTree
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from shapely.geometry import Point
from isometric_logratio import IsometricLogRatio


class GradeBlockReliabilityAssessment(object):
    """
    @brief Evaluate the reliability of grade-block mean estimates.
           The run method returns a result dictionary that contains measures
           of sampling fairness and geochemical consensus for the blasthole
           assay samples within a selective mining unit (grade-block).
    @details See description in "Measuring sampling fairness and geochemical
             consensus for blasthole assays within grade-block mining units"
    """
    def __init__(self, config={}):
        #Generate IID samples (random points uniformly distributed in [0,1] x [0,1])
        self.cfg = config
        self.modelled_elements = self.cfg['modelled_elements']
        n = len(self.modelled_elements)
        self.chemical_weights = np.array(self.cfg.get('chemical_weights', [1/n] * n))
        self.handle_breakdown = self.cfg.get('handle_breakdown', False)
        self.diminishing_sets_terminate_early = self.cfg.get('terminate_early', True)
        self.dof = n - 1
        self.dense_xy_pts = self.precompute_iid_samples()

    def polygon_area(self, x, y):
            correction = x[-1] * y[0] - y[-1] * x[0]
            main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
            return 0.5 * np.abs(main_area + correction)

    def precompute_iid_samples(self, npts=16384, randseed=5836):
        np.random.seed(randseed)
        return np.random.rand(npts, 2)

    def gradeblock_bounding_box(self, verts):
        return {'min': np.min(verts, axis=0), 'max': np.max(verts, axis=0)}

    def generate_particles(self, gb_surface_area, gb_vertices, gb_polygon,
                           min_size=2048, randseed=7425):
        #- precomputed IID samples within a unit square
        pts = self.dense_xy_pts
        bbox = self.gradeblock_bounding_box(gb_vertices)
        range = bbox['max'] - bbox['min']
        #- grow the random point set if the sample size does not satisfy
        #  length(pts) * A(gradeblock_surface) / A(gradeblock_window) >= min_size
        gb_window_area = max(range) ** 2
        if (len(pts) *  gb_surface_area / gb_window_area) < min_size:
            new_size = int(1.05 * min_size * (gb_window_area / gb_surface_area))
            np.random.seed(randseed)
            xy = np.random.rand(new_size, 2)
        else:
            xy = pts
        scaled_pts = bbox['min'] + max(range) * xy
        within = [gb_polygon.contains(Point(p)) for p in scaled_pts]
        return scaled_pts[within]

    def compute_sampling_fairness(self, gb_area, gb_vertices, gb_polygon, gb_holes_xy,
                                  gbstats, gbfactors, results, eps=1.0e-12):
        samples_within = len(gb_holes_xy)
        sampling_evaluation_dict = self.cfg.get('sampling_density_evaluation', {})
        attenuation = float(sampling_evaluation_dict.get('attenuation', 0.1))
        baseline = float(sampling_evaluation_dict.get('baseline', 0.3))
        if samples_within <= 1:
            gbstats['hole_sparsity'] = 1.0 - attenuation
            gbfactors['hole_entropies'] = 1.0
            gbfactors['gradeblock_coverage'] = 1.0
        else:
            particles = self.generate_particles(gb_area, gb_vertices, gb_polygon)
            #- build kd-tree using the blasthole coordinates
            #- query distance to nearest hole for each particle
            tree_holes = cKDTree(gb_holes_xy)
            distances, idx_nearest = tree_holes.query(particles[:,:2], k=1)
            #- compute histogram of hole associated with the particles
            #- use class frequency to compute differential entropy
            hole_freq =[]
            for h in range(samples_within):
                hole_freq.append(sum(idx_nearest == h))
            hole_prob = np.array(hole_freq, dtype=float) / np.sum(hole_freq)
            gbstats['hole_sparsity'] = gbstats['area'] / samples_within
            gbfactors['hole_entropies'] = (sum(-hole_prob * np.log2(hole_prob + eps))
                                           / max(np.log2(samples_within), 1))
            #- examine nearest hole distance distribution
            #  a value tending towards 1 indicates inadequate coverage
            d_inter, _ = tree_holes.query(gb_holes_xy, k=2)
            average_hole_spacing = np.median(d_inter[:,1])
            gbfactors['gradeblock_coverage'] = 1 - float(sum(distances >
                                               average_hole_spacing)) / len(particles)

        gradeblock_hole_densities = 100.0 / gbstats['hole_sparsity']
        sampling_densities = 1.0 - baseline * (attenuation ** gradeblock_hole_densities)
        results['factor_sampling_density'] = sampling_densities
        results['score_spatial_confidence'] = (gbfactors['hole_entropies'] *
                np.sqrt(gbfactors['gradeblock_coverage']) * sampling_densities)

    def small_sample_analysis(self, chemistry, weights=np.r_[1.0,0.65,0.35]):
        conflict_score = 0.0
        eta_outliers = 0.0
        valid_assays = chemistry[np.logical_not(np.any(np.isnan(chemistry),axis=1))]
        median = np.median(valid_assays, axis=0)
        med_abs_dev = np.median(np.abs(valid_assays - median), axis=0)
        z = (valid_assays - median) / med_abs_dev
        n_samples = len(valid_assays)
        lambda_ = 9*((2/np.pi)*np.arctan(np.sqrt(n_samples)))**4.0
        w = np.array(weights) / sum(weights)
        s_outliers = np.array([np.dot(w, np.abs(z_ij)) > lambda_ for z_ij in z])
        if sum(s_outliers) > 0:
            r = np.mean(np.abs(valid_assays[s_outliers] - median), axis=0) / median
            eta_outliers = sum(s_outliers > 0) / n_samples
            conflict_score = min(np.log10(1 + lambda_ * np.dot(w,r)), 1.)
        return eta_outliers, conflict_score

    def compute_geochemical_consensus(self, chemistry, gbstats=None, gbfactors=None,
                                      results=None, randseeds=[8365, 3658, 6583, 5836],
                                      pval=0.025, use_ilr=True, handle_breakdown=None):
        '''
        Evaluate chemistry consensus based on assay samples in the batch.
        @return eta_outliers: estimated fraction of outliers contained in the batch
                rd_gmean: geometric mean of robust distance computed only for outliers,
                rd_score: penalty for suprathreshold distortion on the mean estimate,
                f_consensus: reliability based on observed values in `chemistry`,
                details (dict) with keys:
                - n_samples: samples used in robust estimation
                - robust_dist (array): Mahalanbois distance computed using MCD
                - idx_outiers: indices of non-supporting samples
                - x_star (array): raw assay measurements for non-supporting samples
                - ilr_star (array): the same expressed in ILR transformed coordinates
                - center: robust mean estimate
                - t_chi2: critical value based on chi square distribution
                - err_msg: error message if any    
        '''
        idx_undefined = np.any(np.isnan(chemistry), axis=1)
        idx_contains_zero = np.any(chemistry == 0, axis=1)
        valid_assays = np.logical_not(idx_undefined | idx_contains_zero)
        #apply log-ratio transformation to composition data
        x = np.array(IsometricLogRatio().apply(chemistry, valid_assays)) \
            if use_ilr else chemistry
        n_observ, n_features = x.shape
        x, idx_unique = np.unique(x, axis=0, return_index=True)
        n_samples = len(idx_unique)
        assert self.dof == chemistry.shape[1] - 1
        #apply chi square test to MCD robust distances
        t_chi2 = np.sqrt(chi2.ppf(1-pval, self.dof))
        solution_count = 0
        rd_gmean = 0.0
        rd_score = 0.0
        eta_outliers = 0.0
        lb_eta_outliers = 0.0
        lb_rd_score = 0.0
        if handle_breakdown is None:
            handle_breakdown = self.handle_breakdown

        #When len(randseeds) > 1, it evaluates multiple times to reduce the chances of
        #the MCD algorithm getting trapped in a local minimum. Default: disabled.
        warnings.filterwarnings('error')
        for iter, randseed in enumerate(randseeds):
            rd_gmean_ = 0.0
            rd_score_ = 0.0
            eta_outliers_ = 0.0
            idx_outliers = []
            robust_dist = np.empty((0, n_features))
            center = np.empty((0, n_features))
            err_msg = None
            #compute robust distance: sqrt(cov.mahalanobis(x))= sqrt((x-u)'*inv(C)*(x-u))
            try:
                #use MCD robust estimator to detect outliers.
                #implicitly, support_frac >= (n_observ + n_features + 1.)/(2 * n_observ)
                mcd = MinCovDet(support_fraction=None, random_state=randseed)
                cov = mcd.fit(x)
                robust_dist = np.sqrt(cov.dist_)
                if len(robust_dist) == 0:
                    pass
                elif np.linalg.cond(cov.covariance_) >= 1/sys.float_info.epsilon:
                    err_msg = 'covariance matrix is ill-conditioned.'
                elif err_msg is None:
                    #entry condition ensures the covariance matrix is invertible
                    invC = np.linalg.inv(cov.covariance_)
                    robust_dist = np.sqrt(cov.mahalanobis(x))
                    center = np.mean(x[cov.support_], axis=0) if use_ilr else cov.location_
                    idx_outliers = np.where(robust_dist > t_chi2)[0]
                    eta_outliers_ = float(len(idx_outliers)) / len(x)
                    if len(idx_outliers):
                        rd_gmean_ = np.exp(np.sum(np.log(robust_dist[idx_outliers]))
                                  / len(idx_outliers))
                        rd_score_ = min(np.log10(rd_gmean_/ t_chi2), 1.0)
                    #- update output variables
                    solution_count += 1
                    rd_gmean += rd_gmean_
                    rd_score += rd_score_
                    eta_outliers += eta_outliers_
            except (UserWarning, ValueError) as e:
                #handle special cases where the number of observations, or the
                #effective sample size, is too small to make a reasonable assessment
                if np.any(np.std(x, axis=0)==0): #cov matrix is undefined (not full rank)
                    if n_observ == 1:
                        err_msg = 'info: insufficient samples (n=1)'
                    else:
                        err_msg = 'info: no observed variation (n={})'.format(n_observ)
                elif n_samples < n_features:
                    err_msg = 'covariance matrix is not full rank, will be handled.'
            if iter == 0 or solution_count == 1:
                details = {'n_samples': n_samples,
                           'robust_dist': robust_dist,
                           'idx_outliers': idx_outliers,
                           'x_star': chemistry[idx_unique[idx_outliers]],
                           'ilr_star': x[idx_outliers],
                           'center': center,
                           't_chi2': t_chi2, 
                           'err_msg': err_msg}
        #average estimates
        if solution_count > 1:
            normaliser = 1.0 / solution_count
            rd_gmean *= normaliser
            rd_score *= normaliser
            eta_outliers *= normaliser

        f_consensus = (1.0 - eta_outliers) ** rd_score
        warnings.resetwarnings()
        #conditional update to avoid under-reporting
        if n_samples <= 3 * n_features:
            lb_eta_outliers, lb_rd_score = self.small_sample_analysis(
                                           chemistry, weights=self.chemical_weights)
            details['lb_eta_outliers'] = lb_eta_outliers
            details['lb_rd_score'] = lb_rd_score
            ceiling_consensus = (1.0 - lb_eta_outliers) ** lb_rd_score
            if f_consensus > ceiling_consensus:
                eta_outliers = lb_eta_outliers
                rd_score = lb_rd_score
                f_concensus = ceiling_consensus
        elif n_samples > 6 * n_features and handle_breakdown:
            eta_outliers, rd_gmean, rd_score, f_consensus = \
                self.split_sequence_analysis(chemistry[idx_unique],
                x, eta_outliers, rd_gmean, rd_score, randseed)
 
        if isinstance(gbfactors, dict):
            gbfactors['consensus'] = f_consensus
        if isinstance(gbstats, dict):
            gbstats['eta_outliers'] = eta_outliers
            gbstats['rd_gmean'] = rd_gmean
            gbstats['rd_score'] = rd_score
            gbstats['effective_samples'] = details['n_samples']
        if isinstance(results, dict):
            results['score_chemical_consensus'] = f_consensus
        return eta_outliers, rd_gmean, rd_score, f_consensus, details

    def split_sequence_analysis(self, chemistry, x, eta_outliers, rd_gmean, rd_score,
                                randseed, pval=0.025, dominant_component=0):
        '''
        Perform MCD on split-sequences after re-ordering to alleviate misdetection
        and prevent breakdown when the fraction of outliers >~ 40%.
        '''
        #sort chemistry measurements by the dominant component
        npts = len(chemistry)
        permutation = chemistry[:,dominant_component].argsort()
        xp = x[permutation]
        median = np.median(chemistry[:,dominant_component])
        mcd = MinCovDet(support_fraction=None, random_state=randseed)
        t_chi2 = np.sqrt(chi2.ppf(1-pval, self.dof))
        #form split-sets (divide samples into halves: indexed by sLeft=0, sRight=1)
        sLeft, sRight = 0, 1
        reference_rd_gmean = max(rd_gmean, t_chi2)
        offset = 0
        #when offset==0: split as xxxxxx|oooooo
        #when offset==1: split as xxxxxxx|ooooo

        #identify outliers from split-sets that assimilate well with the other half
        split_outliers_assimilate_well = np.r_[False, False]
        #requirement 1: outliers are located closer to the median than inliers
        #- e.g. in instances xxxx**|oooooo and xxxxxx|*ooooo, the outliers "*" are closer to
        #       the median value "|" than the supporting samples "x" and "o", respectively,
        #       whereas in instances *x*xxx|oooooo and xxxxxx|oooo*o, "*" are further away.
        #requirement 2: largest outlier from splitsets[s] assimilates well with splitsets[1-s]
        #- i.e. its robust_dist w.r.t. splitsets[1-s] is less than max(rd_gmean, t_chi2)
        #       or half the robust_dist w.r.t. splitsets[s]
        while all(split_outliers_assimilate_well == False) and offset <= 1:
            split = (npts >> 1) + offset
            splitsets = [xp[:split], xp[split:]]
            splitchemistry = [chemistry[permutation[:split],dominant_component],
                              chemistry[permutation[split:],dominant_component]]
            for s, splitset in enumerate(splitsets):
                cov = mcd.fit(splitset)
                robust_dist = np.sqrt(cov.dist_)
                idx_outliers = ~cov.support_
                n_outliers = sum(idx_outliers)
                if n_outliers == 0:
                    continue
                rd_max = max(robust_dist[idx_outliers])
                #check if a significant misfit (an incongruous sample) is found
                #- strict admission: rd_max >= 2 * reference_rd_mean
                #- conditional admission : rd_max >= (2 - tolerance) * reference_rd_mean
                tolerance = 0.5
                if rd_max >= (2 - tolerance) * reference_rd_gmean:
                    d0 = np.abs(np.average(splitchemistry[s][idx_outliers],
                                weights=robust_dist[idx_outliers]) - median)
                    d1 = np.abs(np.average(splitchemistry[s][cov.support_],
                                weights=robust_dist[cov.support_]) - median)
                    #impose requirement 1
                    if d0 < d1:
                        cov = mcd.fit(np.vstack((splitsets[1-s],
                                                 splitsets[s][idx_outliers])))
                        rd_complement = np.sqrt(cov.dist_[-n_outliers:])
                        evidence_strength = np.log10((rd_max/reference_rd_gmean)**1.5 * (d1/d0)) \
                                            if d0 != 0 else 1
                        #revoke admission if evidence is weak and passage had relied on tolerance
                        if (evidence_strength < 0.3) and (rd_max < 2.5 * reference_rd_gmean):
                            continue
                        relaxation = max(1, min(evidence_strength, 2.5))
                        #impose requirement 2
                        if min(rd_complement) < max(relaxation * reference_rd_gmean, rd_max/2):
                            split_outliers_assimilate_well[s] = True
                            break
            offset += 1
        #apply estimator to diminishing sets:
        #- evict elements with the largest sub-threshold dist in splitsets[s] one-by-one
        #  note: the evicted elements, inliers in splitsets[s], generally correspond to
        #        outliers when merged and compared with elements in splitsets[1-s].
        #  Each time an element from `retain` is discarded, we are reducing the
        #  contamination level, making MCD outlier detection more efficient. The number
        #  of evicted "outliers" (the correction) will be added to the final count (f).
        existing_consensus = (1.0 - eta_outliers) ** rd_score
        robust_dist[idx_outliers] = 0
        retain = np.argsort(robust_dist)
        max_iterations = 8
        max_outlier_evictions = npts >> 2
        stepsize = max(1, int(np.floor(max_outlier_evictions / max_iterations)))
        evict = 1 - stepsize
        local_rd_gmean = []
        local_rd_score = []
        local_eta_outliers = []
        local_consensus = []
        if any(split_outliers_assimilate_well):
            while evict < max_outlier_evictions:
                evict += stepsize
                cov = mcd.fit(np.vstack((splitsets[1-s], splitsets[s][retain[:-evict]])))
                ind_outliers = ~cov.support_
                n_outliers = sum(ind_outliers)
                if n_outliers == 0:
                    continue #no outliers identified
                local_robust_dist = np.sqrt(cov.dist_)
                rd = np.exp(np.sum(np.log(local_robust_dist[ind_outliers])) / n_outliers)
                if rd <= t_chi2:
                    continue #differences are insignificant
                local_rd_gmean.append( rd )
                local_rd_score.append( min(np.log10(local_rd_gmean[-1]/ t_chi2), 1.0) )
                f = (n_outliers + evict) / npts
                local_eta_outliers.append( min(f, 1-f) )
                local_consensus.append( (1.0 - local_eta_outliers[-1]) ** local_rd_score[-1] )
                if local_consensus[-1] <= 0.8 * existing_consensus and self.diminishing_sets_terminate_early:
                    break
            if len(local_consensus) > 0:
                i = np.argmin(local_consensus)
                if local_consensus[i] < existing_consensus:
                    eta_outliers = local_eta_outliers[i]
                    rd_gmean = local_rd_gmean[i]
                    rd_score = local_rd_score[i]
        #else, no change
        f_consensus = (1.0 - eta_outliers) ** rd_score
        return eta_outliers, rd_gmean, rd_score, f_consensus

    def run(self, gb_polygon, gb_holes_xy, gb_chemistry, results, randseeds=[8365]):
        '''
        API for running the required calculations

        @param gb_polygon (shapely.geometry.Polygon): 2D grade-block boundary
        @param gb_holes_xy (numpy.array):  blasthole coordinates, shape: (n,2)
        @param gb_chemistry (numpy.array): assay measurements, shape: (n,m)
        @param results (dict) report the spatial confidence and geochemical
                              consensus scores along with other statistics.
        '''
        #convert shapely polygon to numpy array
        gb_vertices = np.array(gb_polygon.exterior.xy).T
        gb_surface_area = self.polygon_area(gb_vertices[:,0], gb_vertices[:,1])
        #internal containers for gradeblock statistics
        gbstats = {'area':gb_surface_area, 'hole_count':len(gb_holes_xy),
                   'effective_samples':None, 'hole_sparsity':np.nan,
                   'eta_outliers':None, 'rd_gmean':None, 'rd_score':None}
        gbfactors = {'hole_entropies':np.nan, 'gradeblock_coverage':np.nan,
                     'consensus':np.nan}
        #------------------------------------------------------------------------
        #evaluate sample spatial distribution
        self.compute_sampling_fairness(gb_surface_area, gb_vertices, gb_polygon,
                                       gb_holes_xy, gbstats, gbfactors, results)
        #evaluate geochemical variation
        self.compute_geochemical_consensus(
             gb_chemistry, gbstats, gbfactors, results, randseeds)
        #------------------------------------------------------------------------
        #produce a summary
        results['score_overall_reliability'] = results['score_spatial_confidence'] * \
                                               results['score_chemical_consensus']
        results['factor_gradeblock_coverage'] = gbfactors['gradeblock_coverage']
        results['factor_hole_entropies'] = gbfactors['hole_entropies']
        results['stats_hole_sparsity'] = gbstats['hole_sparsity']
        results['stats_rd_gmean'] = gbstats['rd_gmean']
        results['stats_rd_score'] = gbstats['rd_score']
        results['stats_eta_outliers'] = gbstats['eta_outliers']
        results['stats_area'] = gbstats['area']
        results['stats_hole_count'] = gbstats['hole_count']
        results['stats_effective_samples'] = gbstats['effective_samples']
        return results
