#!/bin/bash

NO_CHANGES_MD5=$(perl script/edit_config_ini.pl --config_path=test/projects/allparamtypes/config.ini | sort | md5sum | sed -e 's/ .*//')

NO_CHANGES_EXPECTED="1bcff9742205e422e140ff03b01977ba"

if [[ "$NO_CHANGES_MD5" == "$NO_CHANGES_EXPECTED" ]]; then
	echo "No changes ok";
else
        echo "No changes failed (got: $NO_CHANGES_MD5, expected: $NO_CHANGES_EXPECTED)"
        exit 1
fi

DELETED_DIMENSION_MD5=$(perl script/edit_config_ini.pl --config_path=test/projects/allparamtypes/config.ini --delete_dimension=lognormal | sort | md5sum | sed -e 's/ .*//')

DELETED_DIMENSION_EXPECTED="180e5e01d99c5ec93b9c8e53e3a51a5d"

if [[ "$DELETED_DIMENSION_MD5" == "$DELETED_DIMENSION_EXPECTED" ]]; then
	echo "Deleted dimension ok";
else
        echo "Deleted dimension failed (got; $DELETED_DIMENSION_MD5, expected: $DELETED_DIMENSION_EXPECTED)"
        exit 2
fi

exit 0
