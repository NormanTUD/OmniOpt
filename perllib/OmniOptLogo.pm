use strict;
use warnings;
use Term::ANSIColor;

sub show_cat {
        my @cats = (
<<EOF
     _ _..._ __
    \)`    (` /
     /      `\
    |  d  b   |
    =\  Y    =/--..-="````"-.
      '.=__.-'               `\
         o/                 /\ \
          |                 | \ \   / )
           \    .--""`\    <   \ '-' /
          //   |      ||    \   '---'
     jgs ((,,_/      ((,,___/
Source: http://www.oocities.org/spunk1111/pets.htm
EOF
,
<<EOF
      |\      _,,,---,,_
ZZZzz /,`.-'`'    -.  ;-;;,_
     |,4-  ) )-,_. ,\ (  `'-'
    '---''(_/--'  `-'\_)  Felix Lee 
Source: http://www.oocities.org/spunk1111/pets.htm
EOF
,
<<EOF
           (\_/)
           /. .\
          =\_T_/=
           /   \ .-.
           | _ |/
          /| | |\
          \)_|_(/  _
   jgs    `"" ""` (_)_.-.
                        '-'
Source: http://www.oocities.org/spunk1111/pets.htm
EOF
,
<<EOF
     (\
      ))         )\\
     ((         /  .(
      \\.-"```"'` =_/=
       >  ,       /
       \   )__.\ |
        > / /  ||\\
   jgs  \\ \\  \\ \\
         `" `" `"  `"
Source: http://www.oocities.org/spunk1111/pets.htm
EOF
,
<<EOF
              .-o=o-.
          ,  /=o=o=o=\ .--.
         _|\|=o=O=o=O=|    \
     __.'  a`\=o=o=o=(`\   /  
     '.   a 4/`|.-""'`\ \ ;'`)   .---.
       )   .'  /   .--'  |  /   / .-._)
       '\   _.'   /     /`-;__.' /
     jgs '--.____.\    /--.___.-'
                   `""`
Source: http://www.oocities.org/spunk1111/pets.htm
EOF
,
<<EOF


             *     ,MMM8&&&.            *
                  MMMM88&&&&&    .
                 MMMM88&&&&&&&
     *           MMM88&&&&&&&&
                 MMM88&&&&&&&&
                 'MMM88&&&&&&'
                   'MMM8&&&'      *    
          |\___/|     /\___/\
          )     (     )    ~( .              '
         =\     /=   =\~    /=
           )===(       ) ~ (
          /     \     /     \
          |     |     ) ~   (
         /       \   /     ~ \
         \       /   \~     ~/
  jgs_/\_/\__  _/_/\_/\__~__/_/\_/\_/\_/\_/\_
  |  |  |  |( (  |  |  | ))  |  |  |  |  |  |
  |  |  |  | ) ) |  |  |//|  |  |  |  |  |  |
  |  |  |  |(_(  |  |  (( |  |  |  |  |  |  |
  |  |  |  |  |  |  |  |\)|  |  |  |  |  |  |
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Source: http://www.oocities.org/spunk1111/pets.htm
EOF
,
<<EOF
                  ;,_            ,
                 _uP~"b          d"u,
                dP'   "b       ,d"  "o
               d"    , `b     d"'    "b
              l] [    " `l,  d"       lb
              Ol ?     "  "b`"=uoqo,_  "l
            ,dBb "b        "b,    `"~~TObup,_
          ,d" (db.`"         ""     "tbc,_ `~"Yuu,_
        .d" l`T'  '=                      ~     `""Yu,
      ,dO` gP,                           `u,   b,_  "b7
     d?' ,d" l,                           `"b,_ `~b  "1
   ,8i' dl   `l                 ,ggQOV",dbgq,._"  `l  lb
  .df' (O,    "             ,ggQY"~  , @@@@\@d"bd~  `b "1
 .df'   `"           -=\@QgpOY""     (b  @@@\@P db    `Lp"b,
.d(                  _               "ko "=d_,Q`  ,_  "  "b,
Ql         .         `"qo,._          "tQo,_`""bo ;tb,    `"b,
qQ         |L           ~"QQQgggc,_.,dObc,opooO  `"~~";.   __,7,
qp         t\\io,_           `~"TOOggQV""""        _,dg,_ =PIQHib.
`qp        `Q["tQQQo,_                          ,pl{QOP"'   7AFR`
  `         `tb  '""tQQQg,_             p" "b   `       .;-.`Vl'
             "Yb      `"tQOOo,__    _,edb    ` .__   /`/'|  |b;=;.__
                           `"tQQQOOOOP""`"\\QV;qQObob"`-._`\\_~~-._
                                """"    ._        /   | |oP"\\_   ~\\ ~\\_~\
                                        `~"\\ic,qggddOOP"|  |  ~\\   `\\~-._
                                          ,qP`"""|"   | `\\ `;   `\\   `\
                               _        _,p"     |    |   `\\`;    |    |
       unknown                 "boo,._dP"       `\\_  `\\    `\\|   `\\   ;
                                 `"7tY~'            `\\  `\\    `|_   |
                                                      `~\\  |

Source: https://user.xmission.com/~emailbox/ascii_cats.htm
EOF
,
<<EOF
            _,'|             _.-''``-...___..--';)
           /_ \'.      __..-' ,      ,--...--'''
          <\    .`--'''       `     /'
           `-';'               ;   ; ;
     __...--''     ___...--_..'  .;.'
    (,__....----'''       (,..--''   Felix Lee 

Source: https://user.xmission.com/~emailbox/ascii_cats.htm
EOF
        );

        print $cats[rand @cats];
}

sub show_logo {
    if(rand() >= 0.9) {
        show_cat();
    }

    my $rand = int(rand(3));
    my ($red, $reset) = (color("red"), color("reset"));
    if($rand == 0) {
        print $red;
        print <<EOF;

 ▄██████▄     ▄▄▄▄███▄▄▄▄   ███▄▄▄▄    ▄█   ▄██████▄     ▄███████▄     ███     
 ███    ███  ▄██▀▀▀███▀▀▀██▄ ███▀▀▀██▄ ███  ███    ███   ███    ███ ▀█████████▄ 
 ███    ███  ███   ███   ███ ███   ███ ███▌ ███    ███   ███    ███    ▀███▀▀██ 
 ███    ███  ███   ███   ███ ███   ███ ███▌ ███    ███   ███    ███     ███   ▀ 
 ███    ███  ███   ███   ███ ███   ███ ███▌ ███    ███ ▀█████████▀      ███     
 ███    ███  ███   ███   ███ ███   ███ ███  ███    ███   ███            ███     
 ███    ███  ███   ███   ███ ███   ███ ███  ███    ███   ███            ███     
  ▀██████▀    ▀█   ███   █▀   ▀█   █▀  █▀    ▀██████▀   ▄████▀         ▄████▀   
                                                                                 
EOF
        print $reset;
    } elsif ($rand == 1) {
        print $red;
        print <<EOF;

 ▒█████   ███▄ ▄███▓ ███▄    █  ██▓ ▒█████   ██▓███  ▄▄▄█████▓
 ▒██▒  ██▒▓██▒▀█▀ ██▒ ██ ▀█   █ ▓██▒▒██▒  ██▒▓██░  ██▒▓  ██▒ ▓▒
 ▒██░  ██▒▓██    ▓██░▓██  ▀█ ██▒▒██▒▒██░  ██▒▓██░ ██▓▒▒ ▓██░ ▒░
 ▒██   ██░▒██    ▒██ ▓██▒  ▐▌██▒░██░▒██   ██░▒██▄█▓▒ ▒░ ▓██▓ ░ 
 ░ ████▓▒░▒██▒   ░██▒▒██░   ▓██░░██░░ ████▓▒░▒██▒ ░  ░  ▒██▒ ░ 
 ░ ▒░▒░▒░ ░ ▒░   ░  ░░ ▒░   ▒ ▒ ░▓  ░ ▒░▒░▒░ ▒▓▒░ ░  ░  ▒ ░░   
   ░ ▒ ▒░ ░  ░      ░░ ░░   ░ ▒░ ▒ ░  ░ ▒ ▒░ ░▒ ░         ░    
   ░ ░ ░ ▒  ░      ░      ░   ░ ░  ▒ ░░ ░ ░ ▒  ░░         ░      
       ░ ░         ░            ░  ░      ░ ░
                                                                     
EOF
        print $reset;
    } elsif ($rand == 2) {
        print <<EOF;
                                                                                          
  ,ad8888ba,                                     88    ,ad8888ba,                         
 d8"'    `"8b                                    ""   d8"'    `"8b                 ,d     
d8'        `8b                                       d8'        `8b                88     
88          88  88,dPYba,,adPYba,   8b,dPPYba,   88  88          88  8b,dPPYba,  MM88MMM  
88          88  88P'   "88"    "8a  88P'   `"8a  88  88          88  88P'    "8a   88     
Y8,        ,8P  88      88      88  88       88  88  Y8,        ,8P  88       d8   88     
 Y8a.    .a8P   88      88      88  88       88  88   Y8a.    .a8P   88b,   ,a8"   88,    
  `"Y8888Y"'    88      88      88  88       88  88    `"Y8888Y"'    88`YbbdP"'    "Y888  
                                                                     88                   
                                                                     88                   
EOF
    }

	return;
}

1;
