import numpy as np
import pandas as pd


class TransformationOfLogWithNewFeatures:

    def __init__(self, dataframe_path, is_train_or_test='train'):
        '''
        This class takes the initial dataframe of log and transform it
        to add new features to create a transformed set for the algorithm to detect
        fake users probability.
        :param dataframe: input dataframe path
        :param is_train_or_test: 'train' or 'test'
        '''

        self.dataframe = pd.read_csv(dataframe_path)
        self.is_train_or_test = is_train_or_test
        self.transformed_dataframe = pd.DataFrame()

    @staticmethod
    def __add_no_of_events__(sub_dataframe_provided):
        '''
        It adds a feature to dataframe mentioning the number of events
        each user did on it's arrival on the website.

        :param sub_dataframe_provided: a subset dataframe containing single userid
        log as recorded
        :return:
        '''

        return sub_dataframe_provided.shape[0]

    @staticmethod
    def __add_ratio_of_number_category_interacted__(sub_dataframe_provided,
                                                    total_distinct_categories_in_log=30):

        '''
        It adds the ratio of no. of distinct category user interacted upon total number
        of distinct category interacted by all the user

        :param sub_dataframe_provided: a subset dataframe containing single userid
        log as recorded
        :param total_distinct_categories_in_log : int
        :return:
        '''

        no_of_distinct_categories_interacted = sub_dataframe_provided.groupby('Category').ngroups

        return float(no_of_distinct_categories_interacted / total_distinct_categories_in_log)

    @staticmethod
    def __add_ratio_of_skewness_in_type_of_events__(sub_dataframe_provided):

        '''
        It adds the ratios of click_ad/total_events, send_email/total_events and
        send_sms/total_events. These features will give the information about the typical bot
        behaviour.

        :param sub_dataframe_provided: a subset dataframe containing single userid
        log as recorded
        :return: ratio_click_ad, ratio_send_email, ratio_send_sms
        '''
        ratio_click_ad = 0
        ratio_send_email = 0
        ratio_send_sms = 0

        for idx, data in sub_dataframe_provided.groupby('Event'):
            if idx == 'click_ad':
                ratio_click_ad = data.shape[0] / sub_dataframe_provided.shape[0]
            elif idx == 'send_sms':
                ratio_send_sms = data.shape[0] / sub_dataframe_provided.shape[0]
            elif idx == 'send_email':
                ratio_send_email = data.shape[0] / sub_dataframe_provided.shape[0]
            else:
                continue
        return ratio_click_ad+ratio_send_email+ratio_send_sms

    @staticmethod
    def __add_ratio_of_amount_events_done_more_by_a_typical_bot(sub_dataframe_provided,
                                                                average_events_done_by_typical_real_user=10):

        '''
        It adds the ratio of
        (total events done by a user - average events done by real user)/(average events done by real user)
        This feature is designed to encapsulate the typical bot behaviour, as for a bot this value will tend
        to be more than 1 and for a real user it will lie between 0 to 1 in normal cases.
        :param sub_dataframe_provided: a subset dataframe containing single userid
        log as recorded
        :param average_events_done_by_typical_real_user: the average of the log
        :return:
        '''

        ratio_of_typical_bot_behaviour = \
            (sub_dataframe_provided.shape[0] - average_events_done_by_typical_real_user) \
            / average_events_done_by_typical_real_user

        return ratio_of_typical_bot_behaviour

    def __calculate_helper_values__(self):
        # first we will calculate the total distinct categories in the dataset
        self.total_distinct_categories_in_log = self.dataframe.groupby('Category').ngroups
        # second we will calculate the average_events_done_by_typical_real_user
        number_of_events_done_by_real_user_list = []
        for idx, data in self.dataframe.groupby('UserId'):
            if data['Fake'].all() == 0:
                number_of_events_done_by_real_user_list.append(data.shape[0])
        self.average_events_done_by_typical_real_user = np.mean(number_of_events_done_by_real_user_list)

    def run_transformer_for_train(self):

        self.__calculate_helper_values__()

        counter = 0
        for idx, data in self.dataframe.groupby('UserId'):
            self.transformed_dataframe.loc[counter, 'UserId'] = idx
            self.transformed_dataframe.loc[counter, 'No_of_events'] = self.__add_no_of_events__(data)
            self.transformed_dataframe.loc[counter, 'Ratio_of_categories_interacted'] = \
                self.__add_ratio_of_number_category_interacted__(data,
                                                                 total_distinct_categories_in_log \
                                                                     =self.total_distinct_categories_in_log)

            self.transformed_dataframe.loc[counter, 'Ratio_of_skewness_of_events_to_detect_bot_behaviour'] = \
                self.__add_ratio_of_skewness_in_type_of_events__(data)

            self.transformed_dataframe.loc[counter, 'Ratio_of_amount_events_done_more_by_a_typical_bot'] = \
                self.__add_ratio_of_amount_events_done_more_by_a_typical_bot(data,
                average_events_done_by_typical_real_user=self.average_events_done_by_typical_real_user)
            self.transformed_dataframe.loc[counter, 'Fake'] = data.iloc[0,-1]
            counter+=1

        return self.transformed_dataframe, \
               self.total_distinct_categories_in_log,\
               self.average_events_done_by_typical_real_user

    def run_transformer_for_test(self, total_distinct_categories_in_log, average_events_done_by_typical_real_user):

        counter = 0
        for idx, data in self.dataframe.groupby('UserId'):
            self.transformed_dataframe.loc[counter, 'UserId'] = idx
            self.transformed_dataframe.loc[counter, 'No_of_events'] = self.__add_no_of_events__(data)
            self.transformed_dataframe.loc[counter, 'Ratio_of_categories_interacted'] = \
                self.__add_ratio_of_number_category_interacted__(data,
                                                                 total_distinct_categories_in_log \
                                                                     =total_distinct_categories_in_log)

            self.transformed_dataframe.loc[counter, 'Ratio_of_skewness_of_events_to_detect_bot_behaviour'] = \
                self.__add_ratio_of_skewness_in_type_of_events__(data)

            self.transformed_dataframe.loc[counter, 'Ratio_of_amount_events_done_more_by_a_typical_bot'] = \
                self.__add_ratio_of_amount_events_done_more_by_a_typical_bot(data,
                                                                             average_events_done_by_typical_real_user)
            self.transformed_dataframe.loc[counter, 'Fake'] = data.iloc[0, -1]
            counter += 1

        return self.transformed_dataframe


if __name__ == '__main__':

    a = TransformationOfLogWithNewFeatures(dataframe_path='/home/rohilrg/Documents/hobbist_projects/AreYouABot/data/train/fake_users.csv')

    df = a.run_transformer_for_train()