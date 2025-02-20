#!/usr/bin/bash

if [[ $SHELL =~ "zsh" ]]; then
        ZSHFILE=$HOME/.zshrc
        date=$(date '+%Y-%m-%d_%H:%M:%S')
        cp $ZSHFILE ${ZSHFILE}_backup_$date
        SOURCE_STR="source ~/.omniopt_comp.zsh" 
        cp zsh/comp.sh ~/.omniopt_comp.zsh
        sed -i '/source.*omniopt_comp.zsh/d' $ZSHFILE
        echo $SOURCE_STR >> $ZSHFILE
        echo "Installed OmniOpt-autocompletion. Re-start or re-login shell to make it work."
else
        echo "Cannot install autocompletion for non-zsh-shells"
fi
