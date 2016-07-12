"""
Created on Mon Jul 11

@author: Diya

processing data in pandas data frame

"""

import parsing_xml
import pandas

def proc_text(df = parsing_xml.xml_to_df()):
    df['process'] = df['text'].apply(lambda x: x[0:2])
    return df

