\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bris}[Liam Dalgarno]
\LoadClass[11pt,twoside]{book}

%% PACKAGE IMPORTS

\RequirePackage[a4paper, margin=2cm]{geometry}
\RequirePackage{import}
\RequirePackage{amsmath}
\RequirePackage{hyperref}
\RequirePackage{siunitx}
\RequirePackage{graphicx}

%% USER SETTINGS

\def\brisName#1{\def\brisName{#1}}
\def\brisDepartment#1{\def\brisDepartment{#1}}
\def\brisFaculty#1{\def\brisFaculty{#1}}
\def\brisDegree#1{\def\brisDegree{#1}}
\def\brisTitle#1{\def\brisTitle{#1}}

%% INTERNAL STUFF

\pagestyle{plain}

% semi-accurate word counting
\newcommand{\quickwordcount}[1]{%
  \immediate\write18{texcount -sum=1 -1 -q -merge #1.tex > #1-words.sum }%
  \input{#1-words.sum} %
}

% setup page numbering for each main section
\renewcommand{\frontmatter}{
  \pagestyle{empty}
  \cleardoublepage
  \pagestyle{plain}
  \pagenumbering{roman}
}
\renewcommand{\mainmatter}{
  \cleardoublepage
  \pagestyle{marked}
  \pagenumbering{arabic}
}
\renewcommand{\backmatter}{
  \cleardoublepage
  \pagestyle{marked}
}

% UoB declaration
\newcommand{\makedecl}{
  \newpage
  \thispagestyle{plain}

  \chapter*{Declaration}
  This dissertation is submitted to the University of Bristol in accordance with the
  requirements of the degree of MEng in the Faculty of Engineering. It has not been submitted
  for any other degree or diploma of any examining body. Except where specifically acknowledged,
  it is all the work of the Author.

  \vspace{1.5cm}
  \noindent
  \textsc{
    SIGNED: ....................................................
    DATE: ..........................................
  }
}

% nicely formatted title page
\renewcommand{\maketitle}{
  \newpage
  \thispagestyle{empty}

  \vspace*{\fill}
  \begin{center}
    {\Huge \brisTitle}
    
    \vspace*{7.5mm} 
    {\Large By} \\ 
    \vspace*{7.5mm}
    {\Large\textsc{\brisName}} \\
    
    \vspace*{15mm}
    \includegraphics[width=0.5\textwidth]{assets/bristollogo_colour}
    \vspace*{15mm}

    {\Large\textsc{Department of \brisDepartment}}
    \vspace*{15mm}

    \begin{minipage}{0.7\textwidth}
    {\large A dissertation submitted to the University of Bristol in accordance with the requirements of the degree of \textsc{\brisDegree} in the Faculty of \brisFaculty.}
    \end{minipage}

    \vspace*{7.5mm}
    {\large\textsc{\today}}
    \vspace*{7.5mm}
  \end{center}

  \begin{flushright}
  Word count: \quickwordcount{main}
  \end{flushright}

  \vspace*{\fill}
}
