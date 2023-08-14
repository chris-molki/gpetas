import numpy as np
import datetime

time_format = "%Y-%m-%d %H:%M:%S.%f"


class R00x_california_set_domain:

    def __init__(self, time_origin=None, time_end_training=None, time_end_total=None, bins_Xgrid=50):
        # special events: sequences (locations)
        self.point_E001_Landers = np.array([-116.437, 34.200])
        self.point_E002_HectorMine = np.array([-116.265, 34.603])
        self.point_E003_Ridgecrest = np.array([-117.599, 35.770])

        # X domain
        self.bins_Xgrid = bins_Xgrid
        self.regions_X_domain()

        # time domain (here oriented at Helmstetter 2007)
        self.time_format = time_format
        self.time_format_example = '2005-08-23 00:00:00.0'
        if time_origin is None:
            time_origin = datetime.datetime.strptime('1981-01-01 00:00:00.0', time_format).replace(
                tzinfo=datetime.timezone.utc)
        else:
            time_origin = datetime.datetime.strptime(time_origin, time_format).replace(tzinfo=datetime.timezone.utc)
        if time_end_training is None:
            time_end_training = datetime.datetime.strptime('2005-08-23 00:00:00.0', time_format).replace(
                tzinfo=datetime.timezone.utc)
        else:
            time_end_training = datetime.datetime.strptime(time_end_training, time_format).replace(
                tzinfo=datetime.timezone.utc)
        if time_end_total is None:
            time_end_total = datetime.datetime.strptime('2022-01-01 00:00:00.0', time_format).replace(
                tzinfo=datetime.timezone.utc)
        else:
            time_end_total = datetime.datetime.strptime(time_end_total, time_format).replace(
                tzinfo=datetime.timezone.utc)
        T_borders_all = np.array([0., (time_end_total.timestamp() - time_origin.timestamp()) / (60. * 60. * 24.)])
        T_borders_training = np.array([0., (time_end_training.timestamp() - time_origin.timestamp()) / (
                60. * 60. * 24.)])  # 1/1/1981 to 23/8/2005: 9000 days
        T_borders_testing = np.array([T_borders_training[1], T_borders_all[1]])
        self.T_borders_all = np.copy(T_borders_all)
        self.T_borders_training = np.copy(T_borders_training)
        self.T_borders_testing = np.copy(T_borders_testing)
        self.time_origin = time_origin
        self.t1_time_end_training = time_end_training
        self.t2_time_end_total = time_end_total
        self.list_regions_of_interest = ['R002a', 'R002b', 'R002c', 'R002d', 'R002e', 'R005', 'R006', 'R007a', 'R008']
        #self.list_regions_of_interest = ['R005', 'R006', 'R007a', 'R008']

        # info: Helmstetter 2007 paper
        m1 = 'Helmstetter, 2007: SRL \n ' \
             'training period originally: 1 January 1981 to (1/1/1996)\n' \
             'training period final     : 1 January 1981 to 23/8/2005 (it was extended to this date)\n' \
             'in Helmstetter test period: 1/1/1996 to 8/23/2005 \n magnitude: >=2\n' \
             'Our testing period: 8/23/2005 to 01/01/2022 magnitude >= 3.0'
        m2 = 'We use earthquakes of M ≥ 2 in the Advanced National Seismic System (ANSS) catalog,\n in the time period from 1 January 1981 to 23 August 2005. We selected earthquakes\n within a rectangu- lar area 30.55° < latitude < 43.95° and –126.35° < longitude < –112.15°,\n larger by 1° than the RELM testing area, to avoid finite region size effects.\n'
        m3 = 'Our final model (our forecast for the next five years) is model #21 in table 1, \n , which uses all available data.\n Model #21 in table 1 uses all the available input data, from 1 January 1981 to 23 August 2005\n'
        m4 = 'The resulting declustered catalog has 81,659 “indepen- dent events” \n (“mainshocks” and “background events”), and 75,545 “dependent events” \n (“foreshocks” and “aftershocks”). The parameters of the declustering algorithm \n were adjusted to remove large fluctuations of seismic activity in space and time.\n'
        # print(m1, m2, m3, m4)

    def get_region(self, region_string, time_origin=None, time_end_training=None, time_end_total=None, bins_Xgrid=None):
        region_obj = region_class()
        region_obj.time_format = self.time_format
        region_obj.time_format_example = self.time_format_example
        if hasattr(self, 'polygon_%s' % region_string):
            print('Region object: %s_obj has been created.' % region_string)
            region = getattr(self, 'polygon_%s' % region_string)
            region_obj.X_borders = np.array(
                [[np.min(region[:, 0]), np.max(region[:, 0])], [np.min(region[:, 1]), np.max(region[:, 1])]])
            region_obj.X_borders_original = np.copy(region_obj.X_borders)

            # spatial resolution
            region_obj.binsize = np.diff(region_obj.X_borders) / self.bins_Xgrid
            if bins_Xgrid is None:
                region_obj.bins_Xgrid = int(np.copy(self.bins_Xgrid))
            else:
                region_obj.bins_Xgrid = int(bins_Xgrid)

            # some info
            region_obj.polygon = region
            region_obj.region_string = region_string
            region_obj.case_name = region_string
            region_obj.polygon_CA = getattr(self, 'polygon_CA')
            region_obj.X_borders_CA = np.array(
                [[np.min(region_obj.polygon_CA[:, 0]), np.max(region_obj.polygon_CA[:, 0])],
                 [np.min(region_obj.polygon_CA[:, 1]), np.max(region_obj.polygon_CA[:, 1])]])
        else:
            'WARNING: Region %s is not defined as subregion in California.'

        # time domain
        time_format = self.time_format
        if time_origin is None:
            time_origin = self.time_origin
            region_obj.time_origin = time_origin
            region_obj.time_origin = time_origin
        else:
            time_origin = datetime.datetime.strptime(time_origin, time_format).replace(tzinfo=datetime.timezone.utc)
            region_obj.time_origin = time_origin
            region_obj.time_origin = time_origin
        if time_end_training is None:
            time_end_training = self.t1_time_end_training
            region_obj.t1_time_end_training = time_end_training
        else:
            time_end_training = datetime.datetime.strptime(time_end_training, time_format).replace(
                tzinfo=datetime.timezone.utc)
            region_obj.t1_time_end_training = time_end_training
        if time_end_total is None:
            time_end_total = self.t2_time_end_total
            region_obj.t2_time_end_total = time_end_total
        else:
            time_end_total = datetime.datetime.strptime(time_end_total, time_format).replace(
                tzinfo=datetime.timezone.utc)
            region_obj.t2_time_end_total = time_end_total

        T_borders_all = np.array([0., (time_end_total.timestamp() - time_origin.timestamp()) / (60. * 60. * 24.)])
        T_borders_training = np.array([0., (time_end_training.timestamp() - time_origin.timestamp()) / (
                60. * 60. * 24.)])  # 1/1/1981 to 23/8/2005: 9000 days
        T_borders_testing = np.array([T_borders_training[1], T_borders_all[1]])
        region_obj.T_borders_all = np.copy(T_borders_all)
        region_obj.T_borders_training = np.copy(T_borders_training)
        region_obj.T_borders_testing = np.copy(T_borders_testing)

        return region_obj

    def regions_X_domain(self):
        # regions
        # R001 frame
        self.polygon_R001 = np.array([
            [-118., 36.],
            [-116, 36.],
            [-116, 34.],
            [-118, 34.],
            [-118., 36.]])

        # R002a frame (Landers)
        self.polygon_R002a = np.array([
            [-117.5, 35.],
            [-115.5, 35.],
            [-115.5, 33.],
            [-117.5, 33.],
            [-117.5, 35.]])

        # R002b frame (Landers, Hector Mine, Ridgecrest)
        x_low, x_up = -118, -116
        y_low, y_up = 34., 36.
        self.polygon_R002b = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R002c frame (Landers, Hector Mine, Ridgecrest) 2.5 x 2.5 degrees
        # x_low, x_up = -118.25, -115.75
        # y_low, y_up = 33.75, 36.25
        x_low, x_up = -118.3, -115.7
        y_low, y_up = 33.7, 36.3
        self.polygon_R002c = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R002d frame (Ridgecrest) 2. x 2. degrees
        x_low, x_up = -118.5, -116.5
        y_low, y_up = 34.5, 36.5
        self.polygon_R002d = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R002e frame (Ridgecrest)
        x_low, x_up = -119., -117.
        y_low, y_up = 35., 37.
        self.polygon_R002e = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R005 frame (San Francisco)
        x_low, x_up = -122.5, -120.5
        y_low, y_up = 36.5, 38.5
        self.polygon_R005 = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R006 frame (Los Angelos)
        x_low, x_up = -119., -117.
        y_low, y_up = 33., 35.
        self.polygon_R006 = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R007 frame (Lake Tahoe, Yosimity)
        x_low, x_up = -120., -118.
        y_low, y_up = 37., 39.
        self.polygon_R007 = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R007a frame (Lake Tahoe, Yosimity)
        x_low, x_up = -120.5, -118.5
        y_low, y_up = 37., 39.
        self.polygon_R007a = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R008 frame (Sequoia National Park)
        x_low, x_up = -119.5, -117.5
        y_low, y_up = 35.5, 37.5
        self.polygon_R008 = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # TOTAL CA test area (HE07)
        xlon_min_H07 = -126.35
        xlon_max_H07 = -112.15
        ylat_min_H07 = 30.55
        ylat_max_H07 = 43.95
        self.polygon_CA = np.array([
            [xlon_min_H07, ylat_min_H07],
            [xlon_min_H07, ylat_max_H07],
            [xlon_max_H07, ylat_max_H07],
            [xlon_max_H07, ylat_min_H07],
            [xlon_min_H07, ylat_min_H07]])

        # SOUTHERN CALIFORNIA
        x_low, x_up = -120., -113.
        y_low, y_up = 30., 37.
        self.polygon_RSC7 = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # ROSS22 ZONE: SOUTHERN CALIFORNIA
        x_low, x_up = -122., -113.
        y_low, y_up = 30., 37.5
        self.polygon_R002r = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])


# subclasses
class region_class():
    def __init__(self):
        self.T_borders_all = None
        self.T_borders_training = None
        self.T_borders_testing = None
        self.X_borders = None
        self.X_borders_UTM_km = None
        self.X_borders_original = None
        self.time_origin = None
        self.m0 = None
        self.case_name = 'Rxxx'


class event():
    def __init__(self):
        self.time = None
        self.position = None


class special_events():
    def __init__(self):
        self.time_format = "%Y-%m-%d %H:%M:%S.%f"
        self.Landers = event()
        self.Landers.time = datetime.datetime.strptime('1992-06-28 11:57:34.0', time_format).replace(
            tzinfo=datetime.timezone.utc)
        self.Landers.position = np.array([-116.437, 34.200])
        self.Hector_Mine = event()
        self.Hector_Mine.time = datetime.datetime.strptime('1999-10-16 09:46:44.0', time_format).replace(
            tzinfo=datetime.timezone.utc)
        self.Hector_Mine.position = np.array([-116.265, 34.603])
        self.Ridgecrest = event()
        self.Ridgecrest.time = datetime.datetime.strptime('2019-07-06 03:19:53.0', time_format).replace(
            tzinfo=datetime.timezone.utc)
        self.Ridgecrest.position = np.array([-117.599, 35.770])

    '''
        def(self, region_string)
        
        time_separation = np.array([0., 9000., 14975.])
        bins_Xgrid=50
        
        # regions
        # R001 frame
        square_R001 = np.array([
            [-118., 36.],
            [-116, 36.],
            [-116, 34.],
            [-118, 34.],
            [-118., 36.]])

        # R002a frame (Landers)
        square_R002a = np.array([
            [-117.5, 35.],
            [-115.5, 35.],
            [-115.5, 33.],
            [-117.5, 33.],
            [-117.5, 35.]])

        # R002b frame (Landers, Hector Mine, Ridgecrest)
        x_low, x_up = -118, -116
        y_low, y_up = 34., 36.
        square_R002b = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R002c frame (Landers, Hector Mine, Ridgecrest) 2.5 x 2.5 degrees
        x_low, x_up = -118.25, -115.75
        y_low, y_up = 33.75, 36.25
        square_R002c = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R005 frame (San Francisco)
        x_low, x_up = -122.5, -120.5
        y_low, y_up = 36.5, 38.5
        square_R005 = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R006 frame (Los Angelos)
        x_low, x_up = -119., -117.
        y_low, y_up = 33., 35.
        square_R006 = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # R007 frame (Lake Tahoe, Yosimity)
        x_low, x_up = -120., -118.
        y_low, y_up = 37., 39.
        square_R007 = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])

        # TOTAL CA test area
        xlon_min_H07 = -126.35
        xlon_max_H07 = -112.15
        ylat_min_H07 = 30.55
        ylat_max_H07 = 43.95
        square_CA = np.array([
            [xlon_min_H07, ylat_min_H07],
            [xlon_min_H07, ylat_max_H07],
            [xlon_max_H07, ylat_max_H07],
            [xlon_max_H07, ylat_min_H07],
            [xlon_min_H07, ylat_min_H07]])

        # SOUTHERN CALIFORNIA
        x_low, x_up = -120., -113.
        y_low, y_up = 30., 37.
        square_RSCA = np.array([
            [x_low, y_up],
            [x_up, y_up],
            [x_up, y_low],
            [x_low, y_low],
            [x_low, y_up]])


        # spatial domain calX
        if region_string == 'R001':
            region = np.copy(square_R001)
            region_name = region_string
            cat_name_short = 'CA_' + region_name
        if region_string == 'R002a':
            region = np.copy(square_R002a)
            region_name = region_string
            cat_name_short = 'CA_' + region_name
        if region_string == 'R002b':
            region = np.copy(square_R002b)
            region_name = region_string
            cat_name_short = 'CA_' + region_name
        if region_string == 'R002c':
            region = np.copy(square_R002c)
            region_name = region_string
            cat_name_short = 'CA_' + region_name
        if region_string == 'R005':
            region = np.copy(square_R005)
            region_name = region_string
            cat_name_short = 'CA_' + region_name
        if region_string == 'R006':
            region = np.copy(square_R006)
            region_name = region_string
            cat_name_short = 'CA_' + region_name
        if region_string == 'R007':
            region = np.copy(square_R007)
            region_name = region_string
            cat_name_short = 'CA_' + region_name
        if region_string == 'RSCA':
            region = np.copy(square_RSCA)
            region_name = region_string
            cat_name_short = 'CA_' + region_name
        else:
            self.region = region
            self.X_borders = np.array(
                [[np.min(region[:, 0]), np.max(region[:, 0])], [np.min(region[:, 1]), np.max(region[:, 1])]])

            # spatial resolution
            self.binsize = np.diff(self.X_borders[0, :]) / bins_Xgrid

            # time domain calT: oriented on HE07
            self.T_borders_all = np.array([time_separation[0], time_separation[2]])
            self.T_borders_training = np.array([time_separation[0], time_separation[1]])  # 1/1/1981 to 23/8/2005: 9000 days
            self.T_borders_testing = np.array([self.T_borders_training[1], self.T_borders_all[1]])

            # some info
            self.region_string = region_string
            self.case_name = region_string
            self.time_separation = time_separation
            self.bins_Xgrid = bins_Xgrid
            self.square_CA = square_CA
    
    '''
