#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;

sub debug (@);

my %options = (
        debug => 0,
        csv => 0,
        moveunreadable => 0
);

analyze_args(@ARGV);

main();

sub debug (@) {
        my @args = @_;
        if($options{debug}) {
            foreach (@args) {
                if(ref $_) {
                    warn Dumper $_;
                } else {
                    warn "$_\n";
                }
            }
        }
}

sub main {
        debug "main";
        debug \%options;
            
        my @files = <*>;
        my $i = 0;
        foreach my $file (@files) {
            if($file =~ /^slurm-(\d+).out/) {
                debug "Checking $file";
                my $slurm_id = $1;
                if($i != 0) {
                    $i = 1;
                }
                my $command = "perl analyze_log_file.pl --slurmid=$slurm_id --i=$i ";
                $command .= " --csv " if $options{csv};
                $command .= " --debug " if $options{debug};
                $command .= " --nodb " if $options{nodb};
                $command .= " --redocache " if $options{redocache};
                $command .= qq# || mv "slurm-$slurm_id.out" oldslurmlogs/# if $options{moveunreadable};
                debug $command;
                my $output = qx($command);
                print $output;
                $i++;
            }
        }
}

sub analyze_args {
        my @args = @_;

        foreach (@args) {
                if(/^--debug$/) {
                        $options{debug} = 1;
                } elsif (/^--csv$/) {
                        $options{csv} = 1;
                } elsif (/^--moveunreadable$/) {
                        $options{moveunreadable} = 1;
                } elsif (/^--redocache$/) {
                        $options{redocache} = 1;
                } else {
                        die "Unknown option `$_`";
                }
        }
}
