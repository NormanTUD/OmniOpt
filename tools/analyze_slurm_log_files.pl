#!/usr/bin/perl

use strict;
use warnings;
use Term::ANSIColor;
use Data::Dumper;
use Time::HiRes qw( time );

my $start = time();

my %options = (
    slurmid => undef,
    debug => 0,

);

analyze_params(@ARGV);

sub _help {
        my $exit_code = shift;
        print <<EOF;
Analyzes slurm log files.

Parametesr:

        --slurmid=12345                 Checks slurm-12345.out
        --debug                         Enables debug
EOF
        exit($exit_code);
}

sub analyze_params {
    my @params = @_;

    foreach (@params) {
        if(/^--file=(.*)$/) {
            $options{slurmid} = $1;
        if(/^--debug$/) {
            $options{debug} = 1;
        } else {
            warn "Unknown parameter `$_`\n";
            _help(1);
        }
    }

    #debug Dumper \%options;
    }
}
