## Compile Javascript

* Check that you have the last version of Node and NPM
* Install the required npm modules:

```bash
npm install --save-dev
npm install uglify --save-dev
```

* To compile, run:

```bash
npm run build
```

* Comment everything out in `_js/src/images.js`
* In `_config.yml`, replace
```yaml
replace_img:
  re_img:              <img\s*(?<attrs>.*?)\s*/>
  re_ignore:           (re|data)-ignore
  replacement:         |
    <hy-img root-margin="512px" %{attrs}>
      <noscript><img data-ignore %{attrs}/></noscript>
      <span slot="loading" class="loading"><span class="icon-cog"></span></span>
    </hy-img>
```

by
```yaml
replace_img:
  re_img:              <img\s*(?<attrs>.*?)\s*/>
  re_ignore:           (re|data)-ignore
  replacement:         |
    <hy-img root-margin="512px" %{attrs}><img %{attrs}></hy-img>
```
