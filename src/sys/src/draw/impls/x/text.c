#ifndef lint
static char vcid[] = "$Id: text.c,v 1.11 1995/07/28 04:23:24 bsmith Exp bsmith $";
#endif

#if defined(HAVE_X11)
/*
   This file contains simple code to manage access to fonts, insuring that
   library routines access/load fonts only once
 */

#include "ximpl.h"

int XiInitFonts(DrawCtx_X *);
int XiMatchFontSize(XiFont*,int,int);
int XiLoadFont(DrawCtx_X*,XiFont*);
/*
    XiFontFixed - Return a pointer to the selected font.

    Warning: these fonts are never freeded because they can possibly 
  be shared by several windows 
*/
int XiFontFixed( DrawCtx_X *XBWin,int w, int h,XiFont **outfont )
{
  static XiFont *curfont = 0,*font = 0;
  static int    fw = 0, fh = 0;
  if (!curfont) { XiInitFonts( XBWin );}
  if (w != fw || h != fh) {
    if (!font)	font = (XiFont*) PETSCMALLOC(sizeof(XiFont)); CHKPTRQ(font);
    XiMatchFontSize( font, w, h );
    fw = w;
    fh = h;
    /* if (curfont) ? unload current font ? */
    XiLoadFont( XBWin, font );
  }
  curfont = font;
  *outfont = curfont;
  return 0;
}

/* this is set by XListFonts at startup */
#define NFONTS 20
static struct {
    int w, h, descent;
} nfonts[NFONTS];
static int act_nfonts = 0;

/*
  These routines determine the font to be used based on the requested size,
  and load it if necessary
*/

int XiLoadFont( DrawCtx_X *XBWin, XiFont *font )
{
  char        font_name[100];
  XFontStruct *FontInfo;
  XGCValues   values ;

  (void) sprintf(font_name, "%dx%d", font->font_w, font->font_h );
  font->fnt  = XLoadFont( XBWin->disp, font_name );

  /* The font->descent may not have been set correctly; get it now that
      the font has been loaded */
  FontInfo   = XQueryFont( XBWin->disp, font->fnt );
  font->font_descent   = FontInfo->descent;

  /* Storage leak; should probably just free FontInfo? */
  /* XFreeFontInfo( FontInfo ); */

  /* Set the current font in the CG */
  values.font = font->fnt ;
  XChangeGC( XBWin->disp, XBWin->gc.set, GCFont, &values ) ; 
  return 0;
}

/* Code to find fonts and their characteristics */
int XiInitFonts( DrawCtx_X *XBWin )
{
  char         **names;
  int          cnt, i, j;
  XFontStruct  *info;

  /* This just gets the most basic fixed-width fonts */
  names   = XListFontsWithInfo( XBWin->disp, "?x??", NFONTS, &cnt, &info );
  j       = 0;
  for (i=0; i < cnt; i++) {
    names[i][1]         = '\0';
    nfonts[j].w         = info[i].max_bounds.width ;
    nfonts[j].h         = info[i].ascent + info[i].descent;
    nfonts[j].descent   = info[i].descent;
    if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
    j++;
    if (j >= NFONTS) break;
  }
  act_nfonts    = j;
  if (cnt > 0)  {
    XFreeFontInfo( names, info, cnt );
  }
  /* If the above fails, try this: */
  if (act_nfonts == 0) {
    /* This just gets the most basic fixed-width fonts */
    names   = XListFontsWithInfo( XBWin->disp, "?x", NFONTS, &cnt, &info );
    j       = 0;
    for (i=0; i < cnt; i++) {
        if (strlen(names[i]) != 2) continue;
        names[i][1]         = '\0';
	nfonts[j].w         = info[i].max_bounds.width ;
        /* nfonts[j].w         = info[i].max_bounds.lbearing +
                                    info[i].max_bounds.rbearing; */
        nfonts[j].h         = info[i].ascent + info[i].descent;
        nfonts[j].descent   = info[i].descent;
        if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
        j++;
	if (j >= NFONTS) break;
    }
    act_nfonts    = j;
    XFreeFontInfo( names, info, cnt );
  }
  return 0;
}

int XiMatchFontSize( XiFont *font, int w, int h )
{
  int i,max,imax,tmp;

  for (i=0; i<act_nfonts; i++) {
    if (nfonts[i].w == w && nfonts[i].h == h) {
        font->font_w        = w;
        font->font_h        = h;
        font->font_descent  = nfonts[i].descent;
        return 0;
    }
  }

  /* determine closest fit, per max. norm */
  imax = 0;
  max  = PETSCMAX(PETSCABS(nfonts[0].w - w),PETSCABS(nfonts[0].h - h));
  for (i=1; i<act_nfonts; i++) {
    tmp = PETSCMAX(PETSCABS(nfonts[i].w - w),PETSCABS(nfonts[i].h - h));
    if (tmp < max) {max = tmp; imax = i;}
  }

  /* should use font with closest match */
  font->font_w        = nfonts[imax].w;
  font->font_h        = nfonts[imax].h;
  font->font_descent  = nfonts[imax].descent;
  return 0;
}
#endif
