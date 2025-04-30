from vers import OpenAI

# Initialize the OpenAI client with the provided configuration
client = OpenAI(
    api_key="23b25c73188643a693f7db46b68afc27",  # Required by the library, will be overwritten
    base_url="https://aoai-farm.bosch-temp.com/api/openai/deployments/google-gemini-1-5-flash",
    default_headers={"genaiplatform-farm-subscription-key": "23b25c73188643a693f7db46b68afc27"}
)

# Send a "Hello World" test prompt to the API
response = client.chat.completions.create(
    model="gemini-1.5-flash",  # Required by the endpoint
    n=1,
    messages=[
        {"role": "user", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello World' and confirm the API is working."}
    ]
)

# Extract and print the response
print(response.choices[0].message.content)