#!/bin/bash
# Setup script for Semiotic Analysis Tool environment variables

echo "==================================="
echo "Semiotic Analysis Tool - Setup"
echo "==================================="
echo ""

# Check if .env file exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists. Loading existing configuration..."
    source .env
else
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
fi

echo ""
echo "Environment variable setup:"
echo ""

# Check GOOGLE_API_KEY
if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
    echo "📝 GOOGLE_API_KEY is not set"
    echo "   Get your API key from: https://console.cloud.google.com/apis/credentials"
    echo "   Then add it to the .env file"
else
    echo "✅ GOOGLE_API_KEY is configured"
fi

# Check ENCRYPTION_KEY
if [ -z "$ENCRYPTION_KEY" ] || [ "$ENCRYPTION_KEY" = "your_encryption_key_here" ]; then
    echo "📝 ENCRYPTION_KEY is not set"
    echo "   Generate a key with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
    echo "   Then add it to the .env file"
else
    echo "✅ ENCRYPTION_KEY is configured"
fi

echo ""
echo "To complete setup:"
echo "1. Edit the .env file and add your API keys"
echo "2. Run: source .env"
echo "3. Then you can run: python semiotic_analysis_tool.py"
echo ""
