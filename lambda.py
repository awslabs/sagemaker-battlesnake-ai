import json

print('Loading function')


def start(event, context):
    #print("Received event: " + json.dumps(event, indent=2))
    return "started!"  # Echo back the first key value
    #raise Exception('Something went wrong')
