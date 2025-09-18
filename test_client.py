#!/usr/bin/env python3
"""
Test client for Speech-to-Text API
"""

import requests
import sys
import os
from pathlib import Path

API_BASE_URL = "http://localhost:8000/api/v1/speech"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_supported_languages():
    """Test supported languages endpoint"""
    print("\nTesting supported languages endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/languages")
        print(f"Status: {response.status_code}")
        languages = response.json()
        print(f"Supported languages count: {len(languages)}")
        print(f"First 10 languages: {languages[:10]}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_transcription(audio_file_path, language=None):
    """Test transcription endpoint"""
    print(f"\nTesting transcription with file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found: {audio_file_path}")
        return False
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            data = {}
            if language:
                data['language'] = language
            
            response = requests.post(f"{API_BASE_URL}/transcribe", files=files, data=data)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Transcription: {result['transcription']}")
                print(f"Language: {result['language']}")
                print(f"Processing time: {result['processing_time']:.2f} seconds")
                print(f"File size: {result['file_size']} bytes")
                if result['duration']:
                    print(f"Duration: {result['duration']:.2f} seconds")
            else:
                print(f"Error response: {response.text}")
            
            return response.status_code == 200
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_file_info(audio_file_path):
    """Test file info endpoint"""
    print(f"\nTesting file info with file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found: {audio_file_path}")
        return False
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/file-info", files=files)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Filename: {result['filename']}")
                print(f"Size: {result['size']} bytes")
                print(f"Content type: {result['content_type']}")
                if result['duration']:
                    print(f"Duration: {result['duration']:.2f} seconds")
            else:
                print(f"Error response: {response.text}")
            
            return response.status_code == 200
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    print("=== Speech-to-Text API Test Client ===\n")
    
    # Test basic endpoints
    health_ok = test_health()
    if not health_ok:
        print("Health check failed. Make sure the server is running.")
        sys.exit(1)
    
    languages_ok = test_supported_languages()
    model_info_ok = test_model_info()
    
    # Test with audio file if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else None
        
        file_info_ok = test_file_info(audio_file)
        transcription_ok = test_transcription(audio_file, language)
        
        print(f"\n=== Test Results ===")
        print(f"Health: {'✓' if health_ok else '✗'}")
        print(f"Languages: {'✓' if languages_ok else '✗'}")
        print(f"Model Info: {'✓' if model_info_ok else '✗'}")
        print(f"File Info: {'✓' if file_info_ok else '✗'}")
        print(f"Transcription: {'✓' if transcription_ok else '✗'}")
        
    else:
        print(f"\n=== Test Results ===")
        print(f"Health: {'✓' if health_ok else '✗'}")
        print(f"Languages: {'✓' if languages_ok else '✗'}")
        print(f"Model Info: {'✓' if model_info_ok else '✗'}")
        print("\nTo test transcription, run:")
        print("python test_client.py <audio_file_path> [language_code]")
        print("Example: python test_client.py sample.wav fa")

if __name__ == "__main__":
    main()