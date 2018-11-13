procedure Test is
   type My_Integer is new Integer;
   X : Integer;
   Y : My_Integer := 12;
begin
   X := Integer (Y);
end Test;
