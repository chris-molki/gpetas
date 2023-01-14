import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import gpetas

# some globals
time_format = "%Y-%m-%d %H:%M:%S.%f"
output_dir = "output"
output_dir_tables = "output/tables"
output_dir_figures = "output/figures"
output_dir_data = "output/data"


def init_outdir():
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir_tables):
        os.mkdir(output_dir_tables)
    if not os.path.isdir(output_dir_figures):
        os.mkdir(output_dir_figures)
    if not os.path.isdir(output_dir_data):
        os.mkdir(output_dir_data)


def data_obj__from_catalog_obj(catalog_obj, R_obj, m_min=None, fname_ixymt=None, case_name = None):
    init_outdir()
    catalog = catalog_obj
    time_origin = R_obj.time_origin
    X_borders = R_obj.X_borders
    T_borders_all = R_obj.T_borders_all
    T_borders_training = R_obj.T_borders_training
    T_borders_testing = R_obj.T_borders_testing
    if case_name is None:
        case_name='case01'
        if hasattr(R_obj, 'case_name'):
            case_name = R_obj.case_name
        else:
            time_origin = datetime.datetime.strptime(time_origin, time_format).replace(tzinfo=datetime.timezone.utc)


    # time milli sec into days
    UNIX_timestamp_origin = datetime.datetime.strptime('1970-01-01 00:00:00.0', time_format).replace(
        tzinfo=datetime.timezone.utc)
    t_shift_msec = (time_origin - UNIX_timestamp_origin).total_seconds() * 1000.
    t_days = (catalog.data['origin_time'] - t_shift_msec) / (1000 * 60 * 60 * 24)
    # space
    xlon = catalog.data['longitude']
    ylat = catalog.data['latitude']
    # magnitude
    mag = catalog.data['magnitude']
    # data all
    data = np.array([t_days, mag, xlon, ylat]).T

    # cutting data
    if m_min is None:
        m_min = np.min(mag)
    data = np.array([t_days[mag >= m_min], mag[mag >= m_min], xlon[mag >= m_min], ylat[mag >= m_min]]).T
    # only causal events
    data = data[data[:, 0] >= 0, :]
    min_longitude = X_borders[0, 0]
    max_longitude = X_borders[0, 1]
    min_latitude = X_borders[1, 0]
    max_latitude = X_borders[1, 1]
    idx = np.where(np.logical_and(np.logical_and(data[:, 2] >= min_longitude, data[:, 2] <= max_longitude),
                                  np.logical_and(data[:, 3] >= min_latitude, data[:, 3] <= max_latitude)))
    data = data[idx]

    # check for duplicates in time and jitter (adding some small dt to it)
    if np.sum(np.diff(data[:, 0]) == 0) > 0:
        idx_dupli = np.where(np.diff(data[:, 0]) == 0)
        print('Warning:')
        print('Warning:', np.sum(np.diff(data[:, 0]) == 0), ' Duplicate(s) in the data set.')
        jitter = np.sort(np.diff(data[:, 0]))[np.sort(np.diff(data[:, 0])) > 0][
                     0] / 2.  # half of smallest time difference between events
        print('jitter=', jitter)
        for i in range(len(idx_dupli)):
            print('    Values at idx', idx_dupli[i], 'and', idx_dupli[i] + 1, '(counting from 0 to n-1) are the same.')
            print('    Time:', data[idx_dupli[i], 0], data[idx_dupli[i] + 1, 0], ' days.')
            data[idx_dupli[i] + 1, 0] = np.copy(data[idx_dupli[i] + 1, 0]) + jitter
            print('    NEW Times:', data[idx_dupli[i], 0], data[idx_dupli[i] + 1, 0], ' days.')
        print('NEW number of time duplicates=', np.sum(np.diff(data[:, 0]) == 0))

    # write data file: idx, x_lon, y_lat, mag, time  format for MLE estimation, Gibbs-sampling
    if fname_ixymt is None:
        fname_ixymt = 'comcat_%s_m0_%02i.dat' % (case_name, int(m_min * 10))
    N_lines = len(data[:, 0])
    write_out = np.zeros((N_lines, 5)) * np.nan  # idx,x,y,m,t
    write_out[:, 0] = np.arange(N_lines) + 1  # idx
    write_out[:, 1] = data[:, 2]
    write_out[:, 2] = data[:, 3]
    write_out[:, 3] = data[:, 1]
    write_out[:, 4] = data[:, 0]
    # some info
    print('----------------------------------------------------------------------')
    print('total number of events = ', N_lines)
    print('time origin            = ', time_origin)
    print('Starting time          =', T_borders_all[0], 'time max=', T_borders_all[1])
    print('T_borders all          =', T_borders_all)
    print('|T|                    =', np.diff(T_borders_all).squeeze(), ' days.')
    print('T_borders training     =', T_borders_training)
    print('min event time         =', np.min(write_out[:, 4]))
    print('max event time         =', np.max(write_out[:, 4]))
    print('X_borders              =', X_borders)
    print('|X|=', np.prod(np.diff(X_borders)), 'deg**2')
    print('x lon min', np.min(write_out[:, 1]), 'x lon max', np.max(write_out[:, 1]), 'dx=',
          (np.max(write_out[:, 1]) - np.min(write_out[:, 1])))
    print('y lat min', np.min(write_out[:, 2]), 'y lat max', np.max(write_out[:, 2]), 'dy=',
          (np.max(write_out[:, 2]) - np.min(write_out[:, 2])))
    print('minimum magnitude', np.min(write_out[:, 3]), 'maximum magnitude', np.max(write_out[:, 3]))
    # print('missing magnitude types beyond ML, Mw, Md, mb, M are: ',np.count_nonzero(np.isnan(write_out[:,5])))
    print('Number of identical event times:  ', np.count_nonzero(np.diff(write_out[:, 4]) == 0))
    print('Fname is: ', fname_ixymt)
    # save idx,x,y,m,t format for MLE estimation, Gibbs-sampling
    fout = output_dir_data + '/' + fname_ixymt
    np.savetxt(fout, write_out[:, :5], delimiter='\t', fmt='%.0f\t%.4f\t%.4f\t%.2f\t%.9f')

    # create data_obj for inference: Gibbs sampling, mle
    outdir = output_dir + '/inference_results'
    data_obj = gpetas.some_fun.create_data_obj_from_cat_file(fname=fout,
                                                             X_borders=X_borders,
                                                             T_borders_all=T_borders_all,
                                                             T_borders_training=T_borders_training,
                                                             utm_yes=None,
                                                             T_borders_test=T_borders_testing,
                                                             m0=m_min,
                                                             outdir=outdir,
                                                             case_name=case_name,
                                                             time_origin=time_origin)

    return data_obj


def plot_regions(regions_obj, R_obj=None):
    init_outdir()
    regions = regions_obj

    hf = plt.figure()
    plt.plot(regions.polygon_CA[:, 0], regions.polygon_CA[:, 1], 'k')
    for attribute, value in vars(regions).items():
        if attribute[:10] == 'polygon_R0':
            plt.plot(value[:, 0], value[:, 1], '--k')  # all regions
        if attribute[:7] == 'polygon':
            plt.plot(value[:, 0], value[:, 1], '--k')  # all regions
    plt.plot(regions.point_E001_Landers[0], regions.point_E001_Landers[1], '.k')
    plt.plot(regions.point_E002_HectorMine[0], regions.point_E002_HectorMine[1], '.k')
    plt.plot(regions.point_E003_Ridgecrest[0], regions.point_E003_Ridgecrest[1], '.k')
    # region of interest
    if R_obj is not None:
        plt.plot(R_obj.polygon[:, 0], R_obj.polygon[:, 1], 'r')  # region of interest
    plt.axis('equal')
    plt.ylabel('y, Lat.', fontsize=20)
    plt.xlabel('x, Lon.', fontsize=20)
    # plt.show()
    if R_obj is not None:
        hf.savefig('./' + output_dir_figures + "/F000_H07_area_case_%s.pdf" %R_obj.case_name, bbox_inches='tight')
    else:
        hf.savefig('./' + output_dir_figures + "/F000_H07_area.pdf", bbox_inches='tight')


def plot_regions_with_catalog(catalog_obj, regions_obj=None, R_obj=None, training='yes', label=None, m0=None):
    init_outdir()
    if m0 is not None:
        catalog = catalog_obj.filter('magnitude >= %.2f' % m0)
    else:
        catalog = catalog_obj
    regions = regions_obj
    if regions is not None:
        polygon_CA = regions.polygon_CA

    hf = plt.figure(figsize=(5, 5))
    plt.plot(catalog.data['longitude'], catalog.data['latitude'], '.', markersize=1., color='k')
    if regions_obj is not None:
        for attribute, value in vars(regions).items():
            if attribute[:10] == 'polygon_R0':
                plt.plot(value[:, 0], value[:, 1], '--b')  # all regions
            if attribute[:7] == 'polygon':
                plt.plot(value[:, 0], value[:, 1], '--b')  # all regions
    if R_obj is not None:
        plt.plot(R_obj.polygon[:, 0], R_obj.polygon[:, 1], 'r')  # region of interest
        polygon_CA = R_obj.polygon_CA
    plt.ylabel('y, Lat.', fontsize=20)
    plt.xlabel('x, Lon.', fontsize=20)
    plt.xlim([np.min(polygon_CA[:, 0]), np.max(polygon_CA[:, 0])])
    plt.ylim([np.min(polygon_CA[:, 1]), np.max(polygon_CA[:, 1])])
    plt.locator_params(nbins=4)
    if polygon_CA is not None:
        if training == 'yes':
            plt.text(np.min(polygon_CA[:, 0]) + 1, 33.25,
                     '$N_{\\mathcal{D}}$ = %s \n$m\in$[%.2f,%.2f]'
                     % (catalog.event_count, np.min(catalog.data['magnitude']),
                        np.max(catalog.data['magnitude'])),
                     horizontalalignment='left',
                     verticalalignment='top', fontsize=12, color='dimgray')
        else:
            plt.text(np.min(polygon_CA[:, 0]) + 1, 33.25,
                     '$N_{\mathcal{D}\cup\mathcal{D}^*}$ = %s \n$m\in$[%.2f,%.2f]'
                     % (catalog.event_count, np.min(catalog.data['magnitude']),
                        np.max(catalog.data['magnitude'])),
                     horizontalalignment='left',
                     verticalalignment='top', fontsize=12, color='dimgray')
    # label = '(a)'
    if label is not None:
        plt.text(-129, 44, label, verticalalignment='top')
    # plt.show()
    if R_obj is not None:
        if training == 'yes':
            hf.savefig('./' + output_dir_figures + "/F001_m3_H07_area_training_data_case_%s.pdf" %R_obj.case_name, bbox_inches='tight')
        else:
            hf.savefig('./' + output_dir_figures + "/F001_m3_area_all_data_case_%s.pdf" %R_obj.case_name, bbox_inches='tight')
    else:
        if training == 'yes':
            hf.savefig('./' + output_dir_figures + "/F001_m3_H07_area_training_data.pdf", bbox_inches='tight')
        else:
            hf.savefig('./' + output_dir_figures + "/F001_m3_area_all_data.pdf", bbox_inches='tight')
