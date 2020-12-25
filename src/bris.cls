\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bris}[Liam Dalgarno]
\LoadClass[a4paper,11pt,twoside]{memoir}

%% PACKAGE IMPORTS

\RequirePackage[printonlyused]{acronym}
\RequirePackage[algochapter]{algorithm2e}
\RequirePackage{amsmath}
\RequirePackage{braket}
\RequirePackage{caption}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage{import}
\RequirePackage{listings}
\RequirePackage{lipsum}
\RequirePackage{siunitx}
\RequirePackage{titlesec}
\RequirePackage{todonotes}
\RequirePackage{xcolor}

%% USER SETTINGS

\def\brisName#1{\def\brisName{#1}}
\def\brisDepartment#1{\def\brisDepartment{#1}}
\def\brisFaculty#1{\def\brisFaculty{#1}}
\def\brisDegree#1{\def\brisDegree{#1}}
\def\brisTitle#1{\def\brisTitle{#1}}

%% INTERNAL STUFF

\setsecnumdepth{subsection}
\setcounter{tocdepth}{2}

\renewcommand{\lstlistlistingname}{List of Listings}

\setulmarginsandblock{2cm}{2cm}{*}
\setlrmarginsandblock{2cm}{2cm}{*}
\checkandfixthelayout

% make chapter centralised
\makechapterstyle{centrechap}{%
  \renewcommand{\chapterheadstart}{\vspace*{\beforechapskip}}
  \renewcommand{\chapnamefont}{\normalfont\Huge\bfseries}
  \renewcommand{\chapnumfont}{\normalfont\Huge\bfseries}
  \renewcommand{\chaptername}{\centering Chapter}
  \renewcommand{\printchapternum}{\chapnumfont \thechapter:}
  \renewcommand{\afterchapternum}{}
  \renewcommand{\chaptitlefont}{\normalfont\Huge\bfseries}
  \renewcommand\printchaptertitle[1]{
    \chaptitlefont##1
  }
  \renewcommand\printchapternonum[1]{
    \chaptitlefont\centering##1
  }
  \setlength{\beforechapskip}{-\baselineskip}
  \setlength{\midchapskip}{0pt}
  \setlength{\afterchapskip}{3em}
}
\chapterstyle{centrechap}

% add rule to head/foot
\makepagestyle{ruledplain}
\makeevenhead{ruledplain}{\normalfont\textit{\leftmark}}{}{\normalfont\textit{\rightmark}}
\makeoddhead{ruledplain}{\normalfont\textit{\rightmark}}{}{\normalfont\textit{\leftmark}}
\makeevenfoot{ruledplain}{}{\normalfont\thepage}{}
\makeoddfoot{ruledplain}{}{\normalfont\thepage}{}
\makeheadrule{ruledplain}{\textwidth}{0.5pt}
\makeevenfoot{plain}{}{\normalfont\thepage}{}
\makeoddfoot{plain}{}{\normalfont\thepage}{}

\pagestyle{ruledplain}

% semi-accurate word counting
\newcommand{\quickwordcount}[1]{%
  \immediate\write18{texcount -sum=1 -0 -q -merge #1.tex > #1-words.sum }%
  \input{#1-words.sum} %
}

% UoB declaration
% TODO: add params to this
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
  \begin{titlingpage}
    \vspace*{\fill}
    \begin{center}
      {\Huge \brisTitle}
      
      \vspace*{8mm} 
      {\Large By} \\ 
      \vspace*{8mm}
      {\Large\textsc{\brisName}} \\
      
      \vspace*{16mm}
      \includegraphics[width=0.5\textwidth]{assets/bristollogo_colour}
      \vspace*{16mm}

      {\Large\textsc{Department of \brisDepartment}}
      \vspace*{16mm}

      \begin{minipage}{0.7\textwidth}
      {\large A dissertation submitted to the University of Bristol in accordance with the requirements of the degree of \textsc{\brisDegree} in the Faculty of \brisFaculty.}
      \end{minipage}

      \vspace*{16mm}
      {\large\textsc{\today}}
      \vspace*{8mm}
    \end{center}

    \begin{flushright}
    Word count: \quickwordcount{main}
    \end{flushright}

    \vspace*{\fill}
  \end{titlingpage}
}
