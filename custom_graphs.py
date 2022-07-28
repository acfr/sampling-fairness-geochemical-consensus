"""
MSF-GC. This module implements plotting functions
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
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def truncate_colormap(cmap, minval=0.0, maxval=1.0, gamma=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(minval + (maxval-minval) * np.linspace(0, 1.0, n)**gamma))
    return new_cmap

sequential_colormap=truncate_colormap(cm.YlOrRd, 0.2, 1.0)

def make_polygon(points, fill_rgb, line_rgb=(0,0,0,1), thickness=0.5):
    return Polygon(np.c_[points[:,0], points[:,1]],
                   facecolor=fill_rgb, edgecolor=line_rgb,
                   linewidth=thickness)

def draw_grade_blocks(gb_boundaries, gb_names, gb_fill=None, blastholes_xy=None,
                      relative_label_position=None, alpha=0.9, show=True,
                      gamma=1.0, colormap=None, cbar_title=None, cbar_pad=0.03,
                      cbar_width=0.02, cbar_v0_mult=1.0, cbar_vert_mult=1.0,
                      no_label=False, categorise=False, dot_size=2, fontsz=8):
    '''
    @brief Produce a map where each polygon region is coloured according to gb_fill
    '''
    centroids = []
    patches = []
    max_xy = [-np.inf] * 2
    min_xy = [+np.inf] * 2
    plt.figure(figsize=(12,9))
    ax = plt.gca()
    n = len(gb_boundaries)
    #ensure gb_fill.shape is compatible with gb_boundaries
    supplied_colour_intensity = gb_fill is not None
    if gb_fill is None:
        np.random.seed(4680)
        if categorise:
            if colormap is None:
                colormap = cm.Set1
            prefixes = [x[0] for x in gb_names]
            uniq_prefixes = np.unique(prefixes).tolist()
            gb_fill = [colormap(uniq_prefixes.index(p)) for p in prefixes]
        else:
            gb_fill = 0.2 * np.random.rand(n,3) + 0.8 * np.ones((n,3))
    else:
        intensity = np.array(gb_fill)
        if intensity.ndim == 2 and len(intensity) < n:
            gb_fill = np.tile(intensity[-1], (n,1))
        elif intensity.ndim == 1:
            if len(intensity) == 3:
                gb_fill = np.tile(intensity, (n,1))
            else:
                if intensity.shape[0] < n:
                    intensity = np.tile(intensity[-1], (n))
                if colormap is None:
                    colormap = truncate_colormap(cm.Blues, 0.05, 1.0, gamma, n=256)
                gb_fill = [colormap(int((q / max(intensity)) * 255)) for q in intensity]
    for pts, name, fill in zip(gb_boundaries, gb_names, gb_fill):
        try:
            patches.append(make_polygon(pts, fill))
            centroids.append(np.mean(pts, axis=0))
            max_xy = np.max(np.vstack([max_xy, np.max(pts,axis=0)]), axis=0)
            min_xy = np.min(np.vstack([min_xy, np.min(pts,axis=0)]), axis=0)
        except Exception as e:
            #handle multi-polygon
            pts_ = np.array([], dtype=float).reshape(0,2)
            for i, xy in enumerate(pts):
                patches.append(make_polygon(np.array(xy), fill if i==0 else (1,1,1)))
                pts_ = np.vstack([pts_, xy])
            centroids.append(np.mean(pts_, axis=0))
    pc = PatchCollection(patches, match_original=True, alpha=alpha)
    ax.add_collection(pc)
    ax.set_xlim([min_xy[0], max_xy[0]])
    ax.set_ylim([min_xy[1], max_xy[1]])
    if supplied_colour_intensity and intensity.ndim == 1:
        gb_labels = ['%.3f' % v for v in intensity]
    else:
        gb_labels = gb_names
    if not no_label:
        for pts, name, fill, cxy in zip(gb_boundaries, gb_labels, gb_fill, centroids):
            if relative_label_position is None:
                plt.text(cxy[0], cxy[1], name, horizontalalignment='center', fontsize=fontsz)
            else:
                xpos = min_xy[0] + relative_label_position[0] * (max_xy[0] - min_xy[0])
                ypos = min_xy[1] + relative_label_position[1] * (max_xy[1] - min_xy[1])
                plt.text(xpos, ypos, name, horizontalalignment='center', fontsize=8)
    plt.axis('scaled')
    if cbar_title is not None:
        hdl = plt.scatter([min_xy[0],max_xy[0]], [min_xy[1],max_xy[1]], c=[0,1], vmin=0, vmax=1, cmap=colormap)
        fig = plt.gcf()
        ax = plt.gca()
        vert_adj = ax.get_position().height * 0.5 * (cbar_vert_mult - 1)
        cax = fig.add_axes([ax.get_position().x1+cbar_pad, ax.get_position().y0-vert_adj,
                            cbar_width, ax.get_position().height*cbar_vert_mult])
        cbar = plt.colorbar(cax=cax)
        cbar.mappable.set_clim(0, 1)
        cbar.ax.set_ylabel(cbar_title, rotation=90)
        hdl.set_visible(False)
    if blastholes_xy is not None:
        plt.scatter(blastholes_xy[:,0], blastholes_xy[:,1], s=dot_size, c='k')
    if show:
        plt.show()   

def ternary_coords(c1, c2, c3):
    '''
    @brief Convert commposition vector c=[c1,c2,c3] into canvas coordinates
           t=[tx,ty] assuming sum(c)=1. The point t is bounded by a unit-length
           upright equilateral triangle anchored at [0,0] in the first quadrant.
    @detail Equations for lines parallel to the bottom and left edges are
            Bottom: y=tan(q)*(x-s) if x-intercept (c2 value) is s, with q=pi/3.
            Left:  y=-tan(q)*x+2*(1-a)*sin(q) if c3 value is a.
    '''
    sin60, tan60 = np.sin(np.pi/3), np.tan(np.pi/3)
    x = 0.5 * (c2 * tan60 + 2 * (1 - c3) * sin60) / tan60
    y = tan60 * (x - c2)
    return [x, y]

def draw_ternary_borders(scales=20, labels=['Fe','SiO2','Al2O3'], showgrid=True,
                         cfg={'osx':0.04, 'osy':0.03, 'lw_fine':0.5}):
    '''
    @brief Draw the outline of a ternary plot, including markers and gridlines
    '''
    plt.figure(figsize=(12,8))
    ticks = np.linspace(0,1,scales+1)[1:-1]
    #internal grid lines to join opposite edges (B=bottom, L=left, R=right)
    finewidth = cfg.get('lw_fine',0.5)
    sin60, cos60 = np.sin(np.pi/3), np.cos(np.pi/3)
    gray = (0.5,0.5,0.5)
    if showgrid:
        for t in ticks:
            plt.plot([t,t+(1-t)*cos60], [0,(1-t)*sin60], c=gray, lw=finewidth)     #(B,R)
            plt.plot([t,t-t*cos60], [0,t*sin60], c=gray, lw=finewidth)             #(B,L)
            plt.plot([t*cos60,1-t*cos60], [t*sin60,t*sin60], c=gray, lw=finewidth) #(L,R)
    #add ticks
    ticks10 = np.linspace(0,1,10+1)  #10% markers
    ticks20 = np.linspace(0,1,20+1)  # 5% markers
    ticks100 = np.linspace(0,1,100+1)# 1% markers
    for t in ticks100:
        plt.plot([t,t-0.01*cos60], [0,-0.01*sin60], c=gray, lw=0.5) #B
        plt.plot([0.5-t*cos60,0.5-(t+0.01)*cos60], [(1-t)*sin60,(1-t+0.01)*sin60], c=gray, lw=0.5) #L
        plt.plot([1-t*cos60,1-t*cos60+0.01], [t*sin60,t*sin60], c=gray, lw=0.5) #R
    for t in ticks20:
        plt.plot([t,t-0.02*cos60], [0,-0.02*sin60], c=gray, lw=0.5)
        plt.plot([0.5-t*cos60,0.5-(t+0.02)*cos60], [(1-t)*sin60,(1-t+0.02)*sin60], c=gray, lw=0.5)
        plt.plot([1-t*cos60,1-t*cos60+0.02], [t*sin60,t*sin60], c=gray, lw=0.5)
    for t in ticks10:
        plt.plot([t,t-0.1*cos60], [0,-0.1*sin60], c=gray, lw=1)
        plt.text(t-0.1*cos60+cfg['osx'], -0.1*sin60+cfg['osy'],
                 '{}'.format('%d%%' % (100*t)), size=9,
                 rotation=60, ha='center', va='center')
        plt.plot([0.5-t*cos60,0.5-(t+0.1)*cos60], [(1-t)*sin60,(1-t+0.1)*sin60], c=(0.5,0.5,0.5), lw=1)
        plt.text(0.5-(t+0.1)*cos60, (1-t+0.1)*sin60-1.5*cfg['osy'],
                 '{}'.format('%d%%' % (100*t)), size=9,
                 rotation=300, ha='center', va='center')
        plt.plot([1-t*cos60,1-t*cos60+0.1], [t*sin60,t*sin60], c=gray, lw=1)
        plt.text(1-t*cos60+1.5*cfg['osx'], t*sin60+0.5*cfg['osy'],
                 '{}'.format('%d%%' % (100*t)), size=9,
                 rotation=0, ha='center', va='center')
    #label axes
    plt.text(1+0.05*cos60,-0.05*sin60,labels[1],size=12,weight='bold')
    plt.text(-0.03*len(labels[2]),-0.05*sin60,labels[2],size=12,weight='bold')
    plt.text(0.5,sin60+0.05*sin60,labels[0],size=12,weight='bold')
    #triangle outline
    plt.plot([0,1,0.5,0],[0,0,sin60,0],'k',linewidth=2)
    plt.axis('equal')
    plt.axis('off')
