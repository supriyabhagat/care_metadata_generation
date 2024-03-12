import requests
from base64 import b64decode
import re
from bs4 import BeautifulSoup

def github_files_content(git_repo, access_token=None, branch='main'):
    supported_extensions=["py", "md", "txt", "sql", "json", "yaml","sh"] # "csv"
    api_url = f'https://api.github.com/repos/{git_repo}/git/trees/{branch}?recursive=1'
    headers = {'Authorization': f'token {access_token}'} if access_token else {}

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        files_content = ""

        for item in data.get('tree', []):
            if item['type'] == 'blob':
                file_path = item['path']
                file_extension = file_path.rsplit('.', 1)[-1].lower()

                if file_extension in supported_extensions:
                    file_content = get_github_file_content(git_repo, access_token, file_path, branch)
                    if file_content is not None:
                        files_content += file_path +"\n"+file_content         
                else:
                    print(f"Skipping unsupported file: {file_path}")
        return files_content
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
        

def get_github_file_content(git_repo, access_token, file_path, branch='main'):
    api_url = f'https://api.github.com/repos/{git_repo}/contents/{file_path}?ref={branch}'
    headers = {'Authorization': f'token {access_token}'}

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'content' in data:
            try:
                # Try decoding with UTF-8
                file_content = b64decode(data['content']).decode('utf-8')
            except UnicodeDecodeError:
                # If decoding with UTF-8 fails, try decoding with Latin-1
                file_content = b64decode(data['content']).decode('latin-1')
            return file_content
        else:
            raise (f"Error: Unable to find content in response - {data}")
            return None
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
        return None


def check_link_origin(url):
    if re.search(r'github', url, re.IGNORECASE):
        return "github"
    elif re.search(r'atlassian', url, re.IGNORECASE):
        return "atlassian"
    else:
        return "Invalid link"

def get_confluence_page_content(confluence_url, page_id, confluence_token):
    api_url = f"https://{confluence_url}/wiki/rest/api/content/{page_id}?expand=body.storage"
    headers = {
        'Authorization': f'Basic {confluence_token}',
        'Content-Type': 'application/json',
    }

    response = requests.get(api_url, headers=headers)

    
    if response.status_code == 200:
        data = response.json()
        html_content = data.get('body', {}).get('storage', {}).get('value', '')

        # Use BeautifulSoup to extract text content from HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text()

        return text_content
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
