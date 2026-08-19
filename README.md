# variationalform.github.io
Git Pages - web site for <http://variationalform.github.io>

``` bash
  583  git add .gitignore 
  591  git add courses_FML_pdf
  592  git status
  593  git commit -m 'message'
  598  git subtree push --prefix courses_FML_pdf origin gh-pages
  602  git add *
  604  git push
```

use this for subfolders
```
git subtree push --prefix courses_FML_pdf origin gh-pages
```
taken from <https://blog.raw.pm/en/deploying-subfolder-github-pages> on 10 jan 2023.



Correcting commit message after pushing... Example:

```
git commit -m 'Amendmants, corrections and updates for August 2026 re-run'
git push
# 'Amendments' needs correcting
git commit --amend -m 'Amendments, corrections and updates for August 2026 re-run'
git push --force

```

