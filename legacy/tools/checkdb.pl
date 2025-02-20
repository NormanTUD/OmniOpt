#!/usr/bin/perl

use strict;
use warnings;
use Term::ANSIColor;
use Data::Dumper;

sub debug (@);

my %options = (
        project => undef,
        minpercenttoexit => 99,
        debug => 0,
        projectdir => undef
);

analyze_args(@ARGV);

main();

sub main {
        debug "main";
        my $exit = get_job_data();
        debug("Exiting with $exit");
        exit($exit);
}

sub get_job_data {
        debug "get_job_data";
        my $command = 'python3 script/getnumberofjobsdone.py --project='.$options{project}.' 2>&1';
        my $string = execute_command($command);

        my %jobdata = (
                jobsindb => 1,
                maxevals => 9999999999999999999999999999999
        );

        foreach my $regextype (qw/jobsindb maxevals/) {
            if($string =~ m#$regextype: (\d+)#) {
                $jobdata{$regextype} = $1;
            }
        }

        $jobdata{percentagedone} = $jobdata{maxevals} / $jobdata{jobsindb};

        debug Dumper \%jobdata;

        if($jobdata{percentagedone} >= $options{minpercenttoexit}) {
                return 1;
        } else {
                return 0;
        }
}

sub execute_command {
        my $command = shift;
        debug $command;
        my $output = qx($command);
        return $output;
}

sub analyze_args {
        foreach my $arg (@_) {
            if($arg =~ /^--debug$/) {
                $options{debug} = 1;
            } elsif ($arg =~ /^--project=(.*)$/) {
                $options{project} = $1;
            } elsif ($arg =~ /^--projectdir=(.*)$/) {
                $options{projectdir} = $1;
            } elsif ($arg =~ /^--minpercenttoexit=(\d+)$/) {
                $options{minpercenttoexit} = $1;
            } else {
                die("Unknown parameter `$arg`");
            }
        }

        die "Please use '--project=projectname'!" unless $options{project};
}

sub debug (@) {
        if($options{debug}) {
                foreach (@_) {
                        warn color("blue")."$_".color("reset")."\n";
                }
        }
}
