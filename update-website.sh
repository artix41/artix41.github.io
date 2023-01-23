bundle exec jekyll build
git add _site
git commit -am "Update website"
git subtree push --prefix _site git@github.com:artix41/artix41.github.io.git master
