import json

print('Loading function')


def start(event, context):
    #print("Received event: " + json.dumps(event, indent=2))
    return {
        "statusCode": 200,
        "body": json.dumps('Snake started')
    }
    #raise Exception('Something went wrong')

def move(event, context):
    #print("Received event: " + json.dumps(event, indent=2))
    return {
        "statusCode": 200,
        "body": json.dumps('Snake moved')
    }
    #raise Exception('Something went wrong')
