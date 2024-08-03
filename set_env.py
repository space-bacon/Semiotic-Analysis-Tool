import os

def set_environment_variable(var_name, var_value):
    os.environ[var_name] = var_value
    print(f"Environment variable {var_name} set.")

def main():
    # Prompt user for each environment variable
    google_api_key = input("Enter your Google API key: ")
    encryption_key = input("Enter your encryption key (32-byte base64 encoded): ")
    
    # Set environment variables
    set_environment_variable("GOOGLE_API_KEY", google_api_key)
    set_environment_variable("ENCRYPTION_KEY", encryption_key)

if __name__ == "__main__":
    main()
