"""
Git Push Helper Script
======================
Helper utility to link and push this local Git repository to GitHub.

Usage:
  python git_push_helper.py --remote-url https://github.com/<your-username>/<your-repo-name>.git
"""

import argparse
import sys
import os

try:
    import dulwich.porcelain as porcelain
    from dulwich.repo import Repo
except ImportError:
    print("Error: dulwich library required. Run: pip install dulwich")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Push local repository to remote GitHub URL")
    parser.add_argument('--remote-url', type=str, required=True, help="Your public GitHub repository URL")
    parser.add_argument('--branch', type=str, default="main", help="Target branch (default: main)")
    args = parser.parse_args()

    repo_path = os.path.dirname(os.path.abspath(__file__))
    repo = Repo(repo_path)
    
    print(f"Configuring remote origin: {args.remote_url}...")
    try:
        # Check existing remotes
        config = repo.get_config()
        config.set(('remote', 'origin'), 'url', args.remote_url.encode('utf-8'))
        config.write_to_path()
        print("Remote origin set successfully.")
    except Exception as e:
        print(f"Notice during remote config: {e}")

    print(f"\nTo push your commit, you can run:")
    print(f"  git remote add origin {args.remote_url}")
    print(f"  git branch -M {args.branch}")
    print(f"  git push -u origin {args.branch}")
    print("\nOr if using GitHub CLI:")
    print(f"  gh repo create <repo-name> --public --source=. --remote=origin --push")

if __name__ == '__main__':
    main()
