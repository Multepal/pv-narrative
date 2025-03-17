<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:tei="http://www.tei-c.org/ns/1.0">

    <xsl:output method="text" encoding="UTF-8"/>

    <!-- Root template -->
    <xsl:template match="/">
        <xsl:apply-templates select="//tei:text/tei:body"/>
    </xsl:template>

    <!-- Process main text body only -->
    <xsl:template match="tei:body">
        <xsl:apply-templates/>
    </xsl:template>

    <xsl:template match="tei:head">
        <xsl:text>&lt;PARTE&gt;</xsl:text>
        <xsl:apply-templates/>
        <!-- <xsl:value-of select="."/> -->
    </xsl:template>

    <xsl:template match="tei:seg[@type='tasik-header']">
        <xsl:text>&lt;CAPITULO&gt;</xsl:text>
        <xsl:apply-templates/>
    </xsl:template>


    <xsl:template match="tei:ref"></xsl:template>
    <xsl:template match="tei:hi"></xsl:template>
    <xsl:template match="tei:pb"></xsl:template>
    <xsl:template match="tei:lb"></xsl:template>
    <xsl:template match="tei:note"></xsl:template>

</xsl:stylesheet>
