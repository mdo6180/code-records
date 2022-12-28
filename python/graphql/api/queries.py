from ariadne import ObjectType


payload = [
    {
        "username": "mdo9",
        "firstname": "Minh Quan",
        "lastname": "Do"
    },
    {
        "username": "mdo8",
        "firstname": "Minh Quan",
        "lastname": "Do"
    },
    {
        "username": "mdo7",
        "firstname": "Minh Ha",
        "lastname": "Do"
    },
]


query = ObjectType("Query")

@query.field("platform")
def resolve_platform(obj, info):
    request = info.context
    user_agent = request.headers.get("user-agent", "guest")
    return f"{user_agent}"

@query.field("users")
def resolve_users(*_, username=None, firstname=None, lastname=None):
    output_payload = None

    if username != None:
        if output_payload == None:
            output_payload = [user for user in payload if user["username"] == username]
        else:
            output_payload = [user for user in output_payload if user["username"] == username]

    if lastname != None:
        if output_payload == None:
            output_payload = [user for user in payload if user["lastname"] == lastname]
        else:
            output_payload = [user for user in output_payload if user["lastname"] == lastname]
    
    if firstname != None:
        if output_payload == None:
            output_payload = [user for user in payload if user["firstname"] == firstname]
        else:
            output_payload = [user for user in output_payload if user["firstname"] == firstname]
    
    if output_payload == None:
        return payload

    return output_payload


user = ObjectType("User")

@user.field("username")
def resolve_username(user, *_):
    return user["username"]

@user.field("firstname")
def resolve_firstname(user, *_):
    return user["firstname"]

@user.field("lastname")
def resolve_lastname(user, *_):
    return user["lastname"]