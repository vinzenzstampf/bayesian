import ROOT

ROOT.TH1.SetDefaultSumw2()
ROOT.gStyle.SetOptStat(0)

f1 = ROOT.TFile.Open('output_ntuple_weighted.root', 'read')
f1.cd()
tree = f1.Get('tree')

variables = [
    ('0'              ,  1,  0   ,  1   , 'normalisation'       , 'events', 0),
    ('ele_pt'         , 30,  0   , 50   , 'electron p_{T} (GeV)', 'events', 0),
    ('ele_phi'        , 15, -3.15,  3.15, 'electron #phi'       , 'events', 0),
    ('ele_eta'        , 15, -2.5 ,  2.5 , 'electron #eta'       , 'events', 0),
    ('ele_dxy'        , 30, -0.5 ,  0.5 , 'electron d_{xy} (cm)', 'events', 1),
    ('ele_dz'         , 45,-15   , 15   , 'electron d_{z} (cm)' , 'events', 1),
    ('ele_genPartFlav', 30,  0   , 30   , 'electron gen flavour', 'events', 1),
    ('ele_iso'        , 30,  0   ,  2   , 'electron iso'        , 'events', 1),
    ('z_eta'          , 15, -2.5 ,  2.5 , 'Z #eta'              , 'events', 1),
    ('z_pt'           , 30,  0   ,100   , 'Z p_{T} (GeV)'       , 'events', 1),
    ('z_mass'         , 30, 80   ,110   , 'Z mass (GeV)'        , 'events', 1),
    ('z_phi'          , 15, -3.15,  3.15, 'Z #phi'              , 'events', 1),
    ('rho'            , 15,  0   , 60   , '#rho (GeV)'          , 'events', 1),
    ('ngvtx'          , 70,  0   , 70   , '#PV'                 , 'events', 1),
    ('weight'         ,100,  0   , 1   , 'FR weight'           , 'events', 1),
#     ('ele_id'         , 30, 0, 50, 'electron d_{z} (cm)' , 'events', 1),
]

for var in variables:

    histo = ROOT.TH1F(var[0], '', var[1], var[2], var[3])
    histo.GetXaxis().SetTitle(var[4])
    histo.GetYaxis().SetTitle(var[5])
    histo.SetTitle('')

    data       = histo.Clone() ; data      .SetName('data_%s'      %var[0])
    prompt     = histo.Clone() ; prompt    .SetName('prompt_%s'    %var[0])
    nonprompt  = histo.Clone() ; nonprompt .SetName('nonprompt_%s' %var[0]) 
    nonprompt1 = histo.Clone() ; nonprompt1.SetName('nonprompt1_%s'%var[0]) 

    c1 = ROOT.TCanvas('c1', '', 700, 700)

    tree.Draw('%s>>data_%s'      %(var[0], var[0]), 'target==1')
    tree.Draw('%s>>prompt_%s'    %(var[0], var[0]), 'target==1 & (ele_genPartFlav==1 | ele_genPartFlav==15 | ele_genPartFlav==22)')
    tree.Draw('%s>>nonprompt_%s' %(var[0], var[0]), '(target==0)*(weight/(1.-weight))') # meant to mimic the data, take it all
    tree.Draw('%s>>nonprompt1_%s'%(var[0], var[0]), '-1. * (target==0 & (ele_genPartFlav==1 | ele_genPartFlav==15 | ele_genPartFlav==22))*(weight/(1.-weight))') # meant to work as 'prompt' subtraction

    nonprompt.Add(nonprompt1)

    data     .SetMarkerStyle(8)
    prompt   .SetFillColor(ROOT.kBlue)
    nonprompt.SetFillColor(ROOT.kOrange-2)

    data     .SetLineColor(ROOT.kBlack)
    prompt   .SetLineColor(ROOT.kBlack)
    nonprompt.SetLineColor(ROOT.kBlack)

    ths = ROOT.THStack('ths','')
    ths.Add(prompt)
    ths.Add(nonprompt)

    ths.Draw('HIST')

    # draw uncertainties
    errors = histo.Clone()
    for hh in ths.GetHists():
        errors.Add(hh)
    errors.SetFillStyle(3344)
    errors.SetFillColor(ROOT.kGray+2)
    errors.Draw('E2 same')

    data.Draw('EP SAME')

    ths.GetXaxis().SetTitle(var[4])
    ths.GetYaxis().SetTitle(var[5])
    ths.GetYaxis().SetTitleOffset(1.5)
    ths.SetMaximum(max([x.GetMaximum() for x in [data, prompt, nonprompt]]))
#     ths.GetYaxis().SetRangeUser(ths.GetMinimum(), max([x.GetMaximum() for x in [data, prompt, nonprompt]]))
    ths.SetTitle('')

    expectation = prompt.Clone()
    expectation.Add(nonprompt)
    ks = data.KolmogorovTest(expectation)

    leg = ROOT.TLegend(.5,.6,.88,.88)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.AddEntry(prompt   , 'prompt/conversions', 'f')
    leg.AddEntry(nonprompt, 'fakes'             , 'f')
    leg.AddEntry(data     , 'observed'          , 'ep')
    leg.AddEntry(0        , 'KS prob %.4f' %ks  , '')
    leg.Draw('same')

    ROOT.gPad.Update()

    ROOT.gPad.SaveAs('%s_stack.pdf' %var[0])
    ROOT.gPad.SaveAs('%s_stack.png' %var[0])

    nonprompt_tight = nonprompt.Clone() ; nonprompt_tight.SetName('nonprompt_tight_%s' %var[0]) 
    nonprompt_loose = nonprompt.Clone() ; nonprompt_loose.SetName('nonprompt_loose_%s' %var[0]) 
    nonprompt_weigh = nonprompt.Clone() ; nonprompt_weigh.SetName('nonprompt_weigh_%s' %var[0]) 

    nonprompt_tight.SetMinimum(0.)
    nonprompt_loose.SetMinimum(0.)
    nonprompt_weigh.SetMinimum(0.)

    tree.Draw('%s>>nonprompt_tight_%s'%(var[0], var[0]), ' (target==1) & !(ele_genPartFlav==1 | ele_genPartFlav==15 | ele_genPartFlav==22)')
    tree.Draw('%s>>nonprompt_loose_%s'%(var[0], var[0]), ' (target==0) & !(ele_genPartFlav==1 | ele_genPartFlav==15 | ele_genPartFlav==22)')
    tree.Draw('%s>>nonprompt_weigh_%s'%(var[0], var[0]), '((target==0) & !(ele_genPartFlav==1 | ele_genPartFlav==15 | ele_genPartFlav==22)) * (weight/(1.-weight))')
    
    nonprompt_tight.SetLineColor(ROOT.kBlue)
    nonprompt_loose.SetLineColor(ROOT.kRed)
    nonprompt_weigh.SetLineColor(ROOT.kOrange)

    nonprompt_tight.SetFillColor(ROOT.kBlue)
    nonprompt_loose.SetFillColor(ROOT.kRed)
    nonprompt_weigh.SetFillColor(ROOT.kOrange)

    nonprompt_tight.SetFillStyle(3345)
    nonprompt_loose.SetFillStyle(3354)
    nonprompt_weigh.SetFillStyle(3395)

    nonprompt_tight.DrawNormalized('hist')
    nonprompt_loose.DrawNormalized('hist same')
    nonprompt_weigh.DrawNormalized('hist same')

    leg = ROOT.TLegend(.5,.6,.88,.88)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.AddEntry(nonprompt_tight, 'fakes in tight', 'l')
    leg.AddEntry(nonprompt_loose, 'fakes in loose', 'l')
    leg.AddEntry(nonprompt_weigh, 'fakes prediction in tight', 'l')
    leg.Draw('same')

    nonprompt_tight.GetYaxis().SetRangeUser(0., nonprompt_tight.GetMaximum()*1.2)
    nonprompt_loose.GetYaxis().SetRangeUser(0., nonprompt_loose.GetMaximum()*1.2)
    nonprompt_weigh.GetYaxis().SetRangeUser(0., nonprompt_weigh.GetMaximum()*1.2)

    ROOT.gPad.Update()

    ROOT.gPad.SaveAs('%s_shapes.pdf' %var[0])
    ROOT.gPad.SaveAs('%s_shapes.png' %var[0])
    
#     import pdb ; pdb.set_trace()
    
    del c1
    del ths
    del data      
    del prompt    
    del nonprompt 
    del nonprompt1
    del histo

