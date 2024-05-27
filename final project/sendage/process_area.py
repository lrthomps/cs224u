import glob
import numpy as np
import pandas as pd
import pickle


def connected_climbs(ascents):
    i_climb = 1
    i_climber = 0
    
    conn_climbs = ascents['climb_id'].values[[0]]
    conn_climbers = ascents[np.isin(ascents['climb_id'], conn_climbs)]['user'].values
    
    to_continue = True
    num_climbs = 1
    num_climbers = len(conn_climbers)
    while to_continue:
        to_continue = False
        conn_climbs = pd.unique(np.append(conn_climbs,
                                          ascents[np.isin(ascents['user'], conn_climbers)]['climb_id'].values))
        if len(conn_climbs) > num_climbs:
            to_continue = True
            num_climbs = len(conn_climbs)
        conn_climbers = pd.unique(np.append(conn_climbers, 
                                            ascents[np.isin(ascents['climb_id'], conn_climbs)]['user'].values))
        if len(conn_climbers) > num_climbers:
            to_continue = True
            num_climbers = len(conn_climbers)

    return conn_climbs, conn_climbers


def process(climbs, ascents, filename=None):
    ascents_df = pd.DataFrame(ascents)
    ascents_df['climb_id'] = ascents_df['climb_id'].astype(int)
    ascents_df['stars'] = ascents_df['stars'].astype(float)
    ascents_df['beta'] = ascents_df['beta'].fillna('').astype(str)
    ascents_df['comments'] = ascents_df['comments'].fillna('').astype(str)
    ascents_df['style'] = ascents_df['style'].fillna('unknown').astype("category")
    ascents_df['grade'] = ascents_df['grade'].astype("category")
    ascents_df['user'] = ascents_df['user'].astype(str)

    success = False
    bad_ids = []
    while not success:
        try:
            ascents_df['date'] = pd.to_datetime(ascents_df['date'], format='mixed')
        except pd.errors.OutOfBoundsDatetime as e:
            pos = int(str(e).split(' ')[-1])
            bad_ids.append(ascents_df.iloc[pos].name)
            ascents_df = ascents_df.drop(ascents_df.iloc[pos].name)
        else:
            success = True
    if len(bad_ids)>0:
        print(f'Dropped bad dates: {bad_ids}')

    climbs_df = pd.DataFrame(climbs)
    climbs_df['id'] = climbs_df['id'].astype(int)
    climbs_df['slug'] = climbs_df['slug'].astype(str)
    climbs_df['name'] = climbs_df['name'].astype(str)
    climbs_df['type'] = climbs_df['type'].astype("category")
    climbs_df['grade'] = climbs_df['grade'].astype("category")
    for field in ['description', 'areas_0_name',
       'areas_1_name', 'areas_2_name', 'areas_0_slug', 'areas_1_slug',
       'areas_2_slug', 'areas_3_name', 'areas_3_slug']:
        climbs_df[field] = climbs_df[field].fillna('').astype(str)
    climbs_df['area_id'] = climbs_df['area_id'].astype(int)
    climbs_df['grade_id'] = climbs_df['grade_id'].astype(int)
    climbs_df['bolts'] = climbs_df['bolts'].astype(int)
    climbs_df['length'] = climbs_df['length'].astype(float)
    climbs_df = climbs_df.set_index('id')

    aggs = ['min', 'max', 'mean', 'count']
    null_stars = pd.concat([ascents_df['climb_id'], 
                            ascents_df['stars'].isnull()], axis=1).groupby('climb_id')['stars'].sum()
    null_stars.name = 'stars_isna'
    stars_stats = ascents_df[['climb_id', 'stars']].groupby('climb_id')['stars'].agg(aggs)
    stars_stats = stars_stats.rename(columns={k: 'ratings_'+ k for k in aggs})
    climbs_df = pd.concat([climbs_df, stars_stats, null_stars], axis=1)
    
    if filename is not None:
        ascents_df.to_hdf(filename, 'ascents', format="table")
        climbs_df.to_hdf(filename, 'climbs', format="table")
        # boulders.to_hdf(filename, 'boulders', format="table")
        # climbers_df.to_hdf(filename, 'climbers', format="table")
    return climbs_df, ascents_df


def combine_areas(path, date='2023-03-26'):
    climb_dfs = []
    ascent_dfs = []
    for f in glob.glob(f'{path}{date}_*_climbs_ascents.p'):
        area = f[22:-17]
        print(f'Processing {area}')
        climbs, ascents = pickle.load(open(f'{path}{date}_{area}_climbs_ascents.p', 'rb'))
        climbs_df, ascents_df = process(climbs, ascents)
        climb_dfs.append(climbs_df)
        ascent_dfs.append(ascents_df)
    
    print(f'Combining areas')
    ascents_df = pd.concat(ascent_dfs, axis=0, ignore_index=True)
    climbs_df = pd.concat(climb_dfs, axis=0)
    
    climbers_, counts = np.unique(ascents_df['user'], return_counts=True)
    climbers_df = pd.DataFrame({'name': climbers_, '# sends': counts})


    conn_climbs, conn_climbers = connected_climbs(ascents_df)
    print(f'fraction connected climbers {len(conn_climbers)/len(climbers_df)}; '
          f'fraction connected climbs {len(conn_climbs)/len(climbs_df)}')
    climbs_df['connected'] = np.isin(climbs_df.index, conn_climbs)
    climbers_df['connected'] = np.isin(climbers_df['name'], conn_climbers)


    boulders = climbs_df[climbs_df['type']=='boulder']
    bascents_df = ascents_df[np.isin(ascents_df['climb_id'], boulders.index)]
    
    conn_bs, conn_bers = connected_climbs(bascents_df)
    print(len(conn_bers)/len(bascents_df['user'].unique()), len(conn_bs)/len(boulders))
    print(f'fraction connected boulderers {len(conn_bers)/len(bascents_df["user"].unique())}; '
          f'fraction connected boulders {len(conn_bs)/len(boulders)}')
    boulders['connected'] = np.isin(boulders.index, conn_bs)
    climbers_df['b_connected'] = np.isin(climbers_df['name'], conn_bers)

    return climbs_df, ascents_df, climbers_df, boulders


def self_consistent_min(ascents, climbers, climbs, n_min=5):
    n_climbs, n_climbers = len(climbs), len(climbers)
    print(n_climbs, n_climbers)
    while (len(climbs) < n_climbs) or (len(climbers) < n_climbers):
        n_climbs, n_climbers = len(climbs), len(climbers)
        print(n_climbs, n_climbers)
        
        # update ascents
        ascents = ascents[np.isin(ascents['user'], climbers.index)]
        ascents = ascents[np.isin(ascents['climb_id'], climbs.index)]
        
        # update climbers
        climbers_, counts = np.unique(ascents['user'], return_counts=True)
        climbers = climbers.loc[climbers_[counts >= n_min]]
        climbers['# sends'] = counts[counts >= n_min]
        
        # update climbs
        climb_styles = ascents.groupby(['climb_id', 'style'])['user'].count().reset_index().pivot(
            index='climb_id', columns='style', values='user').fillna(0)
        climb_styles['# sends'] = climb_styles.sum(axis=1)
        climb_styles = climb_styles[climb_styles['# sends']>=n_min]
        climbs = climbs.loc[climb_styles.index]
        climbs[climb_styles.columns] = climb_styles
    print(len(climbs), len(climbers))
    return ascents, climbers, climbs