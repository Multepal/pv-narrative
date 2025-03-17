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
        <xsl:apply-templates select="tei:div/tei:head | tei:div//tei:rs"/>
    </xsl:template>

    <!-- Extract RS elements with ANA attribute and their DIV context -->
    <xsl:template match="tei:rs">
        <xsl:value-of select="ancestor::tei:div/tei:head"/>
        <xsl:text>|</xsl:text>
        <xsl:value-of select="ancestor::tei:lg/tei:l/tei:seg[@type='tasik-header']"/>
        <xsl:text>|</xsl:text>
        <xsl:value-of select="@ana"/>
        <xsl:text>
</xsl:text>
    </xsl:template>

    <!-- Ignore everything else -->
    <xsl:template match="*"/>

</xsl:stylesheet>
