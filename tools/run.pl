#!/usr/bin/perl

use strict;
use warnings;

use Data::Dumper;
$Data::Dumper::Sortkeys = 1;

use Getopt::Long;

# SOLL:
# sbatch --cpus-per-task='1' -J 'bm100' --mem-per-cpu='3gb' --ntasks='100' --time='20:00:00' sbatch.sh bm100 100
# IST:
# sbatch --cpus-per-task='1' -J 'bm100' --mem-per-cpu='6gb' --ntasks='102' --time='10:00:00' sbatch.sh bm100 100 

my %options = (
    jobname => {
        switch => '-J',
        value => 'omniopt'
    },
    ntasks => {
        switch => '--ntasks',
        value => '10'
    },
    cpuspertask => {
        switch => '--cpus-per-task',
        value => 1
    },
    time => {
        switch => '--time',
        value => '2:00:00'
    },
    partition => {
        switch => '--partition',
        value => undef
    },
    mempercpu => {
        switch => '--mem-per-cpu',
        value => '6gb'
    },
    numberofworkers => 5,
    projectname => '',
    runsbatch => 1,
    debug => 1,
    dryrun => 0,
    forcesbatch => 0,
    forceproject => 0,
    y => 0,
    autooptions => 0,
    reservation => {
        switch => '--reservation',
        value => undef
    }
);

GetOptions(
    "ntasks=i" => \$options{ntasks}{value},
    "cpuspertask=i" => \$options{cpuspertask}{value},
    "numberofworkers=i" => \$options{numberofworkers},
    "runsbatch=i"  => \$options{runsbatch},

    "projectname=s" => \$options{projectname},
    "mempercpu=s" => \$options{mempercpu}{value},
    "jobname=s"   => \$options{jobname}{value},
    "time=s"   => \$options{time}{value},

    "debug"  => \$options{debug},
    "dryrun" => \$options{dryrun},
    "forcesbatch" => \$options{forcesbatch},
    "partition=s" => \$options{partition}{value},
    "forceproject" => \$options{forceproject},
    "y" => \$options{y},
    "autooptions" => \$options{autooptions},
    "reservation=s" => \$options{reservation}{value}
) or die("Error in command line arguments\n");

debug(\%options);
main();

sub main {
    my $run_string = '';

    if($options{autooptions}) {
        if($options{numberofworkers}) {
            $options{ntasks}{value} = $options{numberofworkers} + 2;
            $options{cpuspertask}{value} = 1;

            debug(\%options);
        } else {
                die("--numberofworkers not set, cannot auto-create options");
        }
    }

    if ($options{runsbatch} || $options{forcesbatch}) {
        if(sbatch_installed() || $options{forcesbatch}) {
            $run_string = 'sbatch ';
            foreach my $this_option (sort { $a cmp $b } keys %options) {
                if(ref $options{$this_option}) {
                    my $equals = '=';
                    if($this_option eq 'jobname') {
                        $equals = ' ';
                    }
                    if((exists($options{$this_option}{value}) && defined($options{$this_option}{value})) || !(exists($options{$this_option}{value}))) {
                        $run_string .= $options{$this_option}{switch}.$equals.strquote($options{$this_option}{value})." ";
                    }
                }
            }
        } else {
            print("runsbatch ignored, because `sbatch` cannot be found!\n");
            $run_string = 'bash ';
        }
    }

    $run_string .= ' sbatch.sh ';
    if($options{projectname}) {
        if(-e "projects/$options{projectname}/config.ini" || $options{forceproject}) {
            $run_string .= $options{projectname}.' ';
        } else {
            die("The project $options{projectname} could not be found!");
        }
    } else {
        die("No project name given!");
    }

    if($options{numberofworkers}) {
        $run_string .= $options{numberofworkers}.' ';
    } else {
        die("No number of workers given!");
    }

    $run_string =~ s#\s{2,}# #g;

    print(("=" x length($run_string))."\n$run_string\n".("=" x length($run_string))."\n");

    if($options{dryrun}) {
        print("$run_string\n");
    } else {
        if(!$options{y}) {
            my $pressed_y = 0;
            while (!$pressed_y) {
                print "Is this correct? Press y or j if so!\n";
                my $input = <>;
                chomp $input;
                if($input eq 'y' || $input eq 'j') {
                        $pressed_y = 1;
                }
            }
        }
        system($run_string);
    }
}

sub debug {
    if($options{debug}) {
        foreach (@_) {
            if(ref $_) {
                print Dumper $_;
            } else {
                print "$_\n";
            }
        }
    }
}

sub sbatch_installed {
    my $whereis = qx(whereis sbatch);
    if($whereis =~ m#^sbatch:\s*$#) {
        return 0;
    } else {
        return 1;
    }
}

sub strquote {
    my $arg = shift;

    $arg =~ s/'/'\\''/g;
    return "'" . $arg . "'";
}
