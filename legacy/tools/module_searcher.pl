#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;
use lib './perllib';
use Env::Modify 'system';
use Term::ANSIColor;
use Digest::MD5 qw/md5_hex/;

my %options = (
        modules_to_load => [],
        python_modules_to_load => [],
        bashcode_to_run => [],
        programs_to_find => [],
        modenv => '',
        dryrun => 0
);

use OmniOptFunctions;

analyze_args(@ARGV);

main();

sub list_with_x_first {
        return if @_ == 1;
        my $i = shift;
        ($_[$i], @_[0..$i-1], @_[$i+1..$#_]);
}

sub permutate {
        return [@_] if @_ <= 1;
        return map {
                my ($f, @r) = list_with_x_first($_, @_);
                map [$f, @$_], permutate(@r);
        } 0..$#_;
}

sub red ($) {
        my $p = shift;
        print color('red bold').$p.color("reset")."\n";
}

sub green ($) {
        my $p = shift;
        print color('green bold').$p.color("reset")."\n";
}

sub faculty {
        my $n = shift;
        return 1 if($n == 0 || $n == 1);
        my $fac = 1;
        foreach (2 .. $n) {
                $fac *= $_;
        }
        return $fac;
}

sub cardinality_of_permutation_set {
        my $number_of_items = shift;
        return faculty($number_of_items);
}


sub main {
        my $i = 1;
        my @permutations = permutate(@{$options{modules_to_load}});
        my $number_of_permutations = scalar @permutations;

        foreach my $shuffled (@permutations) {
                print "\n=========== $i of $number_of_permutations ==================>\n";

                print "Trying permutation: \n".join("\n", map { color("underline blue")."ml $_".color("reset") } @{$shuffled})."\n";;

                my $ok = 1;
                module_load("modenv/$options{modenv}");

                foreach my $ml (@{$shuffled}) {
                        module_load($ml);
                }

                foreach my $program (@{$options{programs_to_find}}) {
                        if(!$ok) {
                                print "\nSkipping program check for $program because it is already doomed to fail.";
                                next;
                        }
                        print "which $program";
                        system("which $program");
                        if(!$? == 0) {
                                red "error $program";
                                $ok = 0;
                        } else {
                                green "$program ok";
                        }
                }

                if($ok) {
                        foreach my $bash_code (@{$options{bashcode_to_run}}) {
                                if(!$ok) {
                                        print "\nSkipping bash-code $bash_code because it is already doomed to fail.";
                                        next;
                                }
                                print "bash_code: $bash_code\n";
                                system("$bash_code");
                                if(!$? == 0) {
                                        red "error $bash_code";
                                        $ok = 0;
                                } else {
                                        green "$bash_code ok";
                                }
                        }
                }

                if($ok) {
                        foreach my $python_import (@{$options{python_modules_to_load}}) {
                                if(!$ok) {
                                        print "\nSkipping python import $python_import because it is already doomed to fail.";
                                        next;
                                }
                                print "python_import: $python_import\n";
                                if (!python_module_is_loadable($python_import)) {
                                        red "error $python_import";
                                        $ok = 0;
                                } else {
                                        green "$python_import ok";
                                }
                        }
                }

                if($ok) {
                        print "\nWorking permutation: \n".join("\n", map { color("underline green")."ml $_".color("reset") } @{$shuffled})."\n";;
                        exit(0);
                }

                $i++;
        }

        die "\n".color("red")."All possible combinations exhausted, nothing was found.".color("reset")."\n";
}

sub fisher_yates_shuffle {
        my $deck = shift;  # $deck is a reference to an array
        my $i = @$deck;
        while ($i--) {
                my $j = int rand ($i+1);
                @$deck[$i,$j] = @$deck[$j,$i];
        }
        return $deck;
}


sub python_module_is_loadable {
        my $module = shift;

        system(qq#python3 -c "import $module"#);

        if($? == 0) {
                return 1;
        } else {
                return 0;
        }
}

sub _help {
        my $exit_code = shift // 0;

        print <<EOF;
Example call:
perl module_searcher.pl --modenv=scs5 --ml="Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4,MongoDB/4.0.3,Python/3.7.4-GCCcore-8.3.0" --python_import=hyperopt --programs_to_find=mongod

Parameters:
--modenv=scs5                                                   Environment to be loaded
--ml=ModuleToLoad                                               Modules that need to be loaded
--ml=ModuleToLoad,ModuleTwoToLoad
--python_import=PythonImport                                    Module that needs to be importable by python
--bash=Code                                                     Run code in bash that needs to exit with exit-code 0
--programs_to_find=ls,mongod                                    The listed programs need to be in the PATH after loading the modules
--dryrun                                                        Don't actually ever run anything, just test the permutation algorithm
--help                                                          This help
EOF

        exit $exit_code;
}

sub analyze_args {
        foreach (@_) {
                if(m#^--ml=(.*)$#) {
                        my $match = $1;
                        if($match =~ m#,#) {
                                push @{$options{modules_to_load}}, split(/,/, $match);
                        } else {
                                push @{$options{modules_to_load}}, $match;
                        }
                } elsif (m#^--python_import=(.*)$#) {
                        my $match = $1;
                        if($match =~ m#,#) {
                                push @{$options{python_modules_to_load}}, split(/,/, $match);
                        } else {
                                push @{$options{python_modules_to_load}}, $1;
                        }
                } elsif (m#^--dryrun#) {
                        $options{dryrun} = 1;
                } elsif (m#^--bash=(.*)$#) {
                        push @{$options{bashcode_to_run}}, $1;
                } elsif (m#^--programs_to_find=(.*)$#) {
                        my $match = $1;
                        if($match =~ m#,#) {
                                push @{$options{programs_to_find}}, split(/,/, $match);
                        } else {
                                push @{$options{programs_to_find}}, $match;
                        }
                } elsif (m#^--modenv=(?:modenv)?(ml|scs5|classic)$#) {
                        if($options{modenv} eq "") {
                                $options{modenv} = $1;
                        } else {
                                die "You cannot specify modenv twice!";
                        }
                } elsif (m#^--help$#) {
                        _help();
                } else {
                        die "Unknown switch $_";
                        _help(1);
                }
        }

        if($options{modenv} eq "") {
                die "Modenv needs to be set!";
        }
}

