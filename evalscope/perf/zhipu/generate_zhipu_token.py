
import time
import jwt
import argparse
# pip install PyJWT
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)
 
    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
 
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate zhipu glm api token, you need install PyJWT with: pip install PyJWT")
    parser.add_argument("--api-key", type=str, required=True, 
                        help="The zhipu api key.")
    parser.add_argument("--exp-seconds", type=int, default=24*60*60,
                        help='The token expiration seconds')
    args = parser.parse_args()
    print('Token generated: %s' % generate_token(args.api_key, args.exp_seconds))
    
