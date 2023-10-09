import boto3


#"cluster1(default)
#access_key = 06XRNJUQVQIJLIXH9W1C
#secret_key = F9k2bt18sNz7YbtgKhbgKfRc75v1QPegE1aZkVC5
#host_base = http://10.198.35.2:80

#"cluster2
#access_key = MHHC4X55E5W41Y9SG5QB
#secret_key = bzYsi5qVX960cRIpfHlf8f2iZDTWQCaEjjDUqxDK
#host_base = http://10.198.235.252:80


class Client(object):
    conf1 = {
        "aws_access_key_id":"06XRNJUQVQIJLIXH9W1C",
        "aws_secret_access_key":"F9k2bt18sNz7YbtgKhbgKfRc75v1QPegE1aZkVC5",
        #"endpoint_url":"http://10.198.35.2:80",  #inner ip
        "endpoint_url":"http://10.198.35.254:80", #outside ip
        "service_name":"s3"
    }

    conf2 = {
        "aws_access_key_id":"MHHC4X55E5W41Y9SG5QB",
        "aws_secret_access_key":"bzYsi5qVX960cRIpfHlf8f2iZDTWQCaEjjDUqxDK",
        #"endpoint_url": "http://10.198.235.252:80", #inner ip
        "endpoint_url": "http://10.198.35.252:80",  #outside ip
        "service_name": "s3"
    }

    conf3 = {
        "aws_access_key_id": "MHHC4X55E5W41Y9SG5QB",
        "aws_secret_access_key":"bzYsi5qVX960cRIpfHlf8f2iZDTWQCaEjjDUqxDK",
        "endpoint_url": "http://10.198.35.254:80",
        "service_name": "s3"
    }

    # craw data for f1 english
    conf4 = {
        "aws_access_key_id": "PCFEQW1AK9N7XQCNYBJX",
        "aws_secret_access_key": "X4Lz7NQaA7zUSiEZ5r0VSrNorKTWYX2scIaZ5Q8F",
        "endpoint_url": "http://10.198.15.254:80",
        "service_name": "s3"
    }

    # pjlab
    conf5 = {
        "aws_access_key_id": "LVOH6VJ8LRDSGS48THAF",
        "aws_secret_access_key" : "QOBb91jD3KTqSl15BEy77xIqePlEoRhNOzlDsZw6",
        "endpoint_url": "http://10.135.3.249:80",
        "service_name": "s3"
    }

    def __init__(self, conf=''):
        self.client_ssd = boto3.client(**Client.conf1)
        self.client_hhd = boto3.client(**Client.conf2)
        self.client_asr2 = boto3.client(**Client.conf3)
        self.client_crawler = boto3.client(**Client.conf4)
        self.client_pjlab = boto3.client(**Client.conf5)


    def get(self, url):

        url = url.strip()

        url2 = url.split("s3://")[-1]

        arr = url2.split('/')

        bucket = arr[0]
        key = '/'.join(arr[1:])

        if "speech_annotations_sensebee" == bucket:
            client = self.client_crawler

        elif "speech_annotations" == bucket:
            client = self.client_hhd
        elif "speech_pre_annotations" == bucket:
            client = self.client_hhd
        elif "speech_manual_annotations" == bucket:
            client = self.client_hhd
        elif 'ASR2' == bucket:
            client = self.client_asr2
        elif "ASR" == bucket:
            client = self.client_ssd
        elif "asr" == bucket:
            client = self.client_pjlab
        else:
            # local file
            with open(url, "rb") as fin:
                b = fin.read()
            return b

        b = client.get_object(Bucket=bucket, Key=key)

        #if bucket == "speech_annotations":
        #    b = self.client_hhd.get_object(Bucket=bucket, Key=key)
        #else:
        #    b = self.client_ssd.get_object(Bucket=bucket, Key=key)

        # print(f'{bucket} --> {key}')
        ## ignore not ok
        return b["Body"].read()

if __name__ == '__main__':
    print('Test boto3')

    conf='placeholder'

    client = Client(conf)

    #A =  client.get('s3://ASR/data_aishell/wav/train/S0078/BAC009S0078W0204.wav') 
    #A =  client.get('ASR/data_aishell/wav/train/S0078/BAC009S0078W0204.wav') 
    #A =  client.get('ASR/WenetSpeech/WenetSpeech/audio_seg/train/podcast/B00024/X0000006153_62342628_S00392.wav') 
    #A =  client.get('s3://ASR/ted/CameronHerold_2009X_55.wav\n') 
    #A =  client.get('s3://asr/crawle/youtube/YSqbIpSgxRc/YSqbIpSgxRc_00000_10_4751.wav') 
    #A =  client.get('asr/crawle/youtube/KjXkDTNrUuM/KjXkDTNrUuM_00001_607_962.wav') 
    A =  client.get('asr/task/speech/emotion/Session1/sentences/wav/Ses01M_impro06/Ses01M_impro06_M023.wav') 
    print(len(A))

    #A = client.get('cluster2:s3://speech_annotations/20210719/audio_book_lianting/万古仙穹/169-251.wav')
    #A = client.get('speech_annotations/20210719/audio_book_lianting/万古仙穹/169-251.wav')
    print(len(A))

    #A = client.get('1024SSD2:s3://ASR2/aishell2/iOS/data/wav/C0001/IC0001W0008.wav')
    #A = client.get('ASR2/aishell2/iOS/data/wav/C0001/IC0001W0008.wav')
    print(len(A))

    #A = client.get('crawler:s3://speech_annotations_sensebee/20221222/youtube/4811468339372629494/UCkH2fkUhM1nM86-j6xrg_sQ-8ZHzBTOBxvw_0003.wav')
    #A = client.get('speech_annotations_sensebee/20221222/youtube/4811468339372629494/UCkH2fkUhM1nM86-j6xrg_sQ-8ZHzBTOBxvw_0003.wav')
    #print(len(A))


