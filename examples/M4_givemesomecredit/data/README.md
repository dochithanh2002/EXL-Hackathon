# Give Me Some Credit data

Kaggle's competition rules do not allow redistributing this data, so it is not in the repository. Accept the
rules on the [competition page](https://www.kaggle.com/c/GiveMeSomeCredit), download
[the data](https://www.kaggle.com/c/GiveMeSomeCredit/data), and put `cs-training.csv` and `cs-test.csv` in this
folder. With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```sh
kaggle competitions download -c GiveMeSomeCredit -p examples/M4_givemesomecredit/data
unzip -j examples/M4_givemesomecredit/data/GiveMeSomeCredit.zip cs-training.csv cs-test.csv -d examples/M4_givemesomecredit/data
```

Check the files from this folder with `shasum -a 256 -c` on:

```
1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac  cs-training.csv
bab363a2a807218d32a51f5fc9668b8be7977795065edd386abc8546abaa5b78  cs-test.csv
```

The variables are described in [`../data_dictionary.md`](../data_dictionary.md).
